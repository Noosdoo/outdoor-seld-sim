"""Counterexamples to a universal 59% horn-reach ceiling."""
from audit_v20_reproduce_20260911 import *

def main():
    H=module('audit20_hold_probe',ROOT/'scripts/step12_notify_v9b_hold.py')
    n=H.v9.WARN_CONFIRM
    frames={k:[(1,0.,0.)] for start in [0,60] for k in range(start,start+n)}
    same=H.warn_fires(frames,True)
    one=H.warn_fires({k:[(1,0.,0.)] for k in range(n)},True)
    separated=H.warn_fires({k:[(1,0. if k<60 else 180.,0.)] for start in [0,60] for k in range(start,start+n)},True)
    assert len(one)==1 and len(same)==1 and len(separated)==2
    dump(OUT/'hold_probe.json',dict(code_sha256=sha(ROOT/'scripts/step12_notify_v9b_hold.py'),confirm_frames=n,refractory_frames=H.v9.REFRACTORY,single_event_fires=one,two_same_direction_events_6s_apart=same,two_opposite_direction_events_6s_apart=separated,issue='Clock updated before expiry check: the old same-direction anchor is renewed even after a long silent gap.',universal_59_percent_ceiling=False))
    print('HOLD_PROBE_CONFIRMED')

if __name__=='__main__':main()
