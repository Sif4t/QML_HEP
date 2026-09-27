"""
Reproduce the up-vs-down flavor-tagging dataset used in arXiv:2609.04367,
originally from Lee, Mulligan, Ploskon, Ringer, Yuan (arXiv:2210.06450),
who generated it with PYTHIA6 process 99: LO DIS  gamma* q -> q.

Modern-Pythia8 equivalent of PYTHIA6 process 99:
    WeakBosonExchange:ff2ff(t:gmZ) = on   (t-channel gamma*/Z exchange, code 211)
    WeakZ0:gmZmode = 1                    (restrict to PURE gamma* -> matches proc 99)

Cuts (from the paper): 25 < Q^2 < 1000 GeV^2 ,  0.1 < y < 0.85

Stored per-particle features: z = pT_i/pT_jet, eta, phi (rescaled to [0,1]),
PID (mapped), q (charge). PID mapping: sort the unique PDG IDs that appear
in the sample and space them by 0.1 centered on 0, e.g. -11 -> -0.1, 11 -> 0.1.

NOTE: this paper does not restate the exact e/p beam energies or the jet
radius R used for clustering the single DIS jet -- those live in the
original dataset paper (arXiv:2210.06450, "eRHIC tune", CTEQ6.1 proton PDF).
Reasonable EIC-like defaults are used below (10 GeV e- x 100 GeV p, R=0.8);
adjust to match your target kinematics.

Requirements:
    pip install pythia8mc
    pip install fastjet
"""

import math
import numpy as np
import pythia8
import fastjet as fj

N_JETS_TARGET = 5000
Q2_MIN, Q2_MAX = 25.0, 1000.0
Y_MIN, Y_MAX = 0.1, 0.85
JET_R = 0.8                 # not specified in the QGNN paper -- adjust as needed
MAX_CONSTIT = 10            # paper truncates flavor jets to 10 hardest particles later,
                             # but keep more here and truncate downstream if you like
E_ELECTRON = 10.0            # GeV, adjust to your target EIC energy setting
E_PROTON = 100.0              # GeV

def build_generator():
    pythia = pythia8.Pythia()
    pythia.readString("Beams:frameType = 2")
    pythia.readString("Beams:idA = 2212")          # proton
    pythia.readString("Beams:idB = 11")             # electron
    pythia.settings.parm("Beams:eA", E_PROTON)
    pythia.settings.parm("Beams:eB", E_ELECTRON)

    # Neutral-current DIS via gamma*/Z t-channel exchange, restricted to pure gamma*
    pythia.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    pythia.readString("WeakZ0:gmZmode = 1")   # pure gamma* == PYTHIA6 process 99

    pythia.settings.parm("PhaseSpace:Q2Min", Q2_MIN)
    # Pythia8 core doesn't expose a direct Q2Max phase-space flag; enforce the
    # upper Q2 and y cuts by filtering generated events (below).

    pythia.readString("SpaceShower:dipoleRecoil = on")
    pythia.readString("SpaceShower:pTmaxMatch = 2")
    pythia.readString("PDF:lepton = off")
    pythia.readString("TimeShower:QEDshowerByL = off")

    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 42")
    pythia.init()
    return pythia

def get_jet_definition():
    return fj.JetDefinition(fj.antikt_algorithm, JET_R)

def compute_x_y_Q2(pythia, event):
    info = pythia.info
    # Pythia8's Info object exposes DIS kinematics directly when the
    # WeakBosonExchange DIS processes are used.
    Q2 = info.Q2Fac() if hasattr(info, "Q2Fac") else None
    # Prefer built-in accessors if available in your Pythia8 version:
    try:
        Q2 = info.Q2()
        x = info.x1() if info.id2() == 11 else info.x2()
        y = info.y() if hasattr(info, "y") else None
    except AttributeError:
        Q2, x, y = None, None, None
    return x, y, Q2

def scattered_lepton_and_final_state(event):
    lepton = None
    hadronic = []
    for p in event:
        if not p.isFinal():
            continue
        if lepton is None and abs(p.id()) == 11:
            lepton = p
            continue
        hadronic.append(p)
    return lepton, hadronic

def phi_wrap01(phi):
    return (phi + math.pi) / (2 * math.pi)

def classify_incoming_quark(event):
    """Flavor label = PDG id of the incoming quark that scatters off gamma*."""
    for p in event:
        if abs(p.status()) == 23 and abs(p.id()) in (1, 2, 3, 4, 5):
            return p.id()
    return None

def main():
    pythia = build_generator()
    jet_def = get_jet_definition()

    up_jets, down_jets = [], []
    n_try = 0

    while len(up_jets) + len(down_jets) < N_JETS_TARGET:
        if not pythia.next():
            continue
        n_try += 1

        x, y, Q2 = compute_x_y_Q2(pythia, pythia.event)
        if Q2 is None or y is None:
            continue
        if not (Q2_MIN <= Q2 <= Q2_MAX):
            continue
        if not (Y_MIN <= y <= Y_MAX):
            continue

        flavor = classify_incoming_quark(pythia.event)
        if flavor is None or abs(flavor) not in (1, 2):
            continue
        label = 1 if abs(flavor) == 2 else 0  # 1 = up-type, 0 = down-type

        lepton, hadronic = scattered_lepton_and_final_state(pythia.event)
        pjs = []
        for p in hadronic:
            if abs(p.id()) in (12, 14, 16):
                continue
            pj = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            pj.set_user_index(p.index())
            pjs.append(pj)

        cs = fj.ClusterSequence(pjs, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())
        if not jets:
            continue
        jet = jets[0]  # LO DIS -> single hard jet
        pt = jet.pt()

        constituents = sorted(jet.constituents(), key=lambda c: -c.pt())
        constituents = constituents[:MAX_CONSTIT]

        rows = []
        for c in constituents:
            idx = c.user_index()
            part = pythia.event[idx]
            z = c.pt() / pt
            eta_c = c.eta()
            phi_c = phi_wrap01(c.phi())
            pid = part.id()
            charge = part.charge()
            rows.append((z, eta_c, phi_c, pid, charge))

        feats = np.array(rows, dtype=object)  # (n_constit, 5) raw PID, remap below
        if label == 1:
            up_jets.append(feats)
        else:
            down_jets.append(feats)

        if n_try % 2000 == 0:
            print(f"tried {n_try} -> up {len(up_jets)}, down {len(down_jets)}")

    pythia.stat()

    # Build PID -> value mapping over the WHOLE sample: unique PDG ids sorted,
    # spaced by 0.1, centered at 0 (e.g. -11 -> -0.1, 11 -> 0.1)
    all_pids = sorted({int(row[3]) for jet in (up_jets + down_jets) for row in jet})
    mid = len(all_pids) / 2.0
    pid_to_val = {pid: 0.1 * (i - mid + 0.5) for i, pid in enumerate(all_pids)}

    def remap(jets):
        out = []
        for jet in jets:
            arr = np.array(
                [[row[0], row[1], row[2], pid_to_val[int(row[3])], row[4]] for row in jet],
                dtype=float,
            )
            out.append(arr)
        return np.array(out, dtype=object)

    np.savez(
        "flavor_dis_jets.npz",
        up_jets=remap(up_jets),
        down_jets=remap(down_jets),
        pid_map=pid_to_val,
    )
    print(f"Saved {len(up_jets)} up-jets and {len(down_jets)} down-jets "
          f"-> flavor_dis_jets.npz  (each jet: array of (z, eta, phi, PID_mapped, q))")

if __name__ == "__main__":
    main()
