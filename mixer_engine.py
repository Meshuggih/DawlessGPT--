# /mnt/data/mixer_engine.py
from typing import List, Optional, Dict
import numpy as np
from fx_processors import SchroederReverb, Delay, Compressor, TruePeakLimiter

class Track:
    def __init__(self, name: str, sample_rate: int):
        self.name = name
        self.sample_rate = sample_rate
        self.volume_db = 0.0
        self.pan = 0.0
        self.is_muted = False
        self.is_soloed = False
        self.reverb_send = 0.0
        self.delay_send = 0.0

class MixerEngine:
    def __init__(self, num_tracks: int, sample_rate: int):
        self.sample_rate = sample_rate
        self.tracks: List[Track] = [Track(f"Track{i}", sample_rate) for i in range(num_tracks)]
        self.reverb_bus = SchroederReverb(sample_rate, room_size=0.8, decay=0.6, mix=1.0)
        t = getattr(self, "cfg_fx_delay_time", "1/8.")
        fb = getattr(self, "cfg_fx_delay_feedback", 0.4)
        mx = getattr(self, "cfg_fx_delay_mix", 0.25)
        bpm = getattr(self, "tempo_bpm", 120.0)
        self.delay_bus  = Delay(sample_rate, feedback=fb, mix=mx, time=t, tempo_bpm=bpm)
        self.master_limiter = TruePeakLimiter(ceiling_db=-0.3, oversample=8)
        self.master_pregain = 1.0
        self._sc = []  # (src_index, dst_index, params)
        self.mono_bass_hz = 150

    def create_sc(self, src_name: str, dst_name: str, depth_db: float=3.0, attack_ms: float=5, release_ms: float=80, shape: str="exp"):
        si = next((i for i,t in enumerate(self.tracks) if t.name.lower()==src_name.lower()), None)
        di = next((i for i,t in enumerate(self.tracks) if t.name.lower()==dst_name.lower()), None)
        if si is None or di is None: return
        self._sc.append((si, di, dict(depth_db=depth_db, attack_ms=attack_ms, release_ms=release_ms, shape=shape)))

    def get_track(self, index: int) -> Track: return self.tracks[index]
    def save_snapshot(self, name: str) -> None: ...
    def recall_snapshot(self, name: str) -> None: ...

    def render_mix(self, track_buffers: List[np.ndarray],
                   sidechain_kick: Optional[np.ndarray] = None,
                   sidechain_sources: Optional[Dict[int, np.ndarray]] = None) -> np.ndarray:
        n = max(len(tb) for tb in track_buffers) if track_buffers else 0
        master_left = np.zeros(n, dtype=np.float32)
        master_right = np.zeros(n, dtype=np.float32)

        if len(self.tracks) != len(track_buffers):
            self.tracks = [Track(f"T{i}", self.sample_rate) for i in range(len(track_buffers))]

        rev_in = np.zeros(n, dtype=np.float32)
        del_in = np.zeros(n, dtype=np.float32)

        for i, buf in enumerate(track_buffers):
            tr = self.tracks[i]
            mono = buf.astype(np.float32)
            # volume piste
            mono *= 10.0**(float(getattr(tr,"volume_db",0.0))/20.0)

            # collecter aux sends
            if getattr(tr, "reverb_send", 0.0) > 0.0:
                rev_in[:len(mono)] += mono * float(tr.reverb_send)
            if getattr(tr, "delay_send", 0.0) > 0.0:
                del_in[:len(mono)] += mono * float(tr.delay_send)

            # sidechain multi-routes
            routes = [(s,d,p) for (s,d,p) in self._sc if d==i]
            if routes:
                g_total = np.ones_like(mono)
                for (si, _, params) in routes:
                    sc_sig = None
                    if sidechain_sources is not None and si in sidechain_sources:
                        sc_sig = sidechain_sources[si]
                    elif sidechain_kick is not None:
                        sc_sig = sidechain_kick
                    if sc_sig is None:
                        continue
                    ratio = 1.0 + max(1.0, params.get("depth_db",3.0)/1.5)
                    comp  = Compressor(self.sample_rate, threshold_db=-24, ratio=ratio,
                                       attack_ms=params.get("attack_ms",5),
                                       release_ms=params.get("release_ms",80))
                    ducked = comp.process(mono, sc_sig[:len(mono)])
                    g = ducked / (mono + 1e-12)
                    shape = params.get("shape","exp")
                    if   shape=="exp": g = g**2
                    elif shape=="log": g = np.sqrt(g)
                    g_total *= g
                mono = mono * g_total

            # pan (equal power)
            pan = max(-1.0, min(1.0, getattr(tr, "pan", 0.0)))
            l_gain = np.cos((pan+1.0)*0.25*np.pi)
            r_gain = np.sin((pan+1.0)*0.25*np.pi)
            left = mono * l_gain
            right = mono * r_gain

            master_left[:len(left)] += left
            master_right[:len(right)] += right

        # traiter busses (mono→stéréo simple)
        rev_out = self.reverb_bus.process(rev_in)
        del_out = self.delay_bus.process(del_in)
        master_left[:len(rev_out)] += rev_out * 0.5
        master_right[:len(rev_out)] += rev_out * 0.5
        master_left[:len(del_out)] += del_out * 0.5
        master_right[:len(del_out)] += del_out * 0.5

        # mono-bass < mono_bass_hz
        fc = int(getattr(self, "mono_bass_hz", 150) or 0)
        stereo = np.stack([master_left, master_right], axis=1)
        if fc > 0:
            rc = 2.0 * np.pi * fc / float(self.sample_rate)
            alpha = rc / (rc + 1.0)
            def lp(sig):
                y = np.zeros_like(sig); acc = 0.0
                for i,x in enumerate(sig):
                    acc = acc + alpha*(x-acc); y[i]=acc
                return y
            lowL, lowR = lp(master_left), lp(master_right)
            lowM = 0.5*(lowL + lowR)
            master_left  = lowM + (master_left  - lowL)
            master_right = lowM + (master_right - lowR)
            stereo = np.stack([master_left, master_right], axis=1)

        # pré-gain (headroom) puis True-Peak limiter (unique ici)
        stereo *= float(self.master_pregain)
        stereo = self.master_limiter.process(stereo)
        return stereo