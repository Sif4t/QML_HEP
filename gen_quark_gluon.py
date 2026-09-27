"""
Reproduce the quark-gluon jet dataset used in arXiv:2609.04367 (and originally
the EnergyFlow `qg_jets` dataset, Komiske/Metodiev/Thaler).

Hard processes (pp, sqrt(s) = 14 TeV):
    qqbar -> Z(->nunubar) + g      (gluon-jet sample)
    qg    -> Z(->nunubar) + q      (quark-jet sample, keep only u,d,s)

Jets: anti-kT R=0.4 (FastJet), built from all final-state particles except
neutrinos. Selection: pT in [500,550] GeV, |eta| < 1.7.
Stored per-particle features: z = pT_i/pT_jet, eta, phi (phi rescaled to [0,1]).

Requirements:
    pip install pythia8mc          # Pythia8 python bindings
    pip install fastjet             # FastJet python bindings
    (or: conda install -c conda-forge pythia8 fastjet)
"""

import math
import numpy as np
import pythia8
import fastjet as fj

N_EVENTS_TARGET = 5000          # jets to keep (paper uses 5000 total)
PT_LO, PT_HI = 500.0, 550.0
ETA_MAX = 1.7
JET_R = 0.4
MAX_CONSTIT = 150

def build_generator():
    pythia = pythia8.Pythia()
    pythia.readString("Beams:idA = 2212")
    pythia.readString("Beams:idB = 2212")
    pythia.readString("Beams:eCM = 14000.")

    # Hard processes: qqbar -> Z g  and  qg -> Z q
    pythia.readString("WeakBosonAndParton:qqbar2gmZg = on")
    pythia.readString("WeakBosonAndParton:qg2gmZq = on")
    # Restrict to pure Z (paper specifies Z, not gamma*/Z interference)
    pythia.readString("WeakZ0:gmZmode = 2")
    # Force Z -> nu nubar only, so it leaves the jet "invisible"
    pythia.readString("23:onMode = off")
    pythia.readString("23:onIfAny = 12 14 16")

    # Roughly bias generation toward the analysis pT window for efficiency
    pythia.readString(f"PhaseSpace:pTHatMin = {PT_LO * 0.8}")
    pythia.readString(f"PhaseSpace:pTHatMax = {PT_HI * 1.2}")

    # Default tunes/showers, hadronization + MPI on (paper: "default tunings")
    pythia.readString("PartonLevel:MPI = on")
    pythia.readString("HadronLevel:Hadronize = on")

    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 42")
    pythia.init()
    return pythia

def get_jet_definition():
    return fj.JetDefinition(fj.antikt_algorithm, JET_R)

def event_to_pseudojets(event):
    pjs = []
    for p in event:
        if not p.isFinal():
            continue
        if abs(p.id()) in (12, 14, 16):   # skip neutrinos
            continue
        pj = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
        pj.set_user_index(p.index())
        pjs.append(pj)
    return pjs

def classify_hard_flavor(event):
    """Return the PDG id of the outgoing colored parton from the 2->2 ME,
    used to label whether this event is a (u,d,s) quark event or a gluon
    event. Walk the hard process (status +-23) entries."""
    ids = []
    for p in event:
        if abs(p.status()) == 23 and p.id() != 23:  # exclude the Z itself
            ids.append(p.id())
    # keep the colored one that isn't the beam remnant
    for pid in ids:
        if pid == 21 or abs(pid) in (1, 2, 3, 4, 5):
            return pid
    return None

def phi_wrap01(phi):
    # map (-pi, pi] -> [0, 1)
    return (phi + math.pi) / (2 * math.pi)

def main():
    pythia = build_generator()
    jet_def = get_jet_definition()

    quark_jets = []
    gluon_jets = []

    n_try = 0
    while len(quark_jets) + len(gluon_jets) < N_EVENTS_TARGET:
        if not pythia.next():
            continue
        n_try += 1

        flavor = classify_hard_flavor(pythia.event)
        if flavor is None:
            continue
        if flavor == 21:
            label = 0  # gluon
        elif abs(flavor) in (1, 2, 3):
            label = 1  # u,d,s quark
        else:
            continue   # discard c,b-initiated events per paper

        pjs = event_to_pseudojets(pythia.event)
        cs = fj.ClusterSequence(pjs, jet_def)
        jets = fj.sorted_by_pt(cs.inclusive_jets())

        for jet in jets:
            pt, eta = jet.pt(), jet.eta()
            if not (PT_LO <= pt <= PT_HI):
                continue
            if abs(eta) >= ETA_MAX:
                continue

            constituents = sorted(jet.constituents(), key=lambda c: -c.pt())
            constituents = constituents[:MAX_CONSTIT]

            z = np.array([c.pt() / pt for c in constituents])
            eta_c = np.array([c.eta() for c in constituents])
            phi_c = np.array([phi_wrap01(c.phi()) for c in constituents])
            feats = np.stack([z, eta_c, phi_c], axis=1)  # (n_constit, 3)

            if label == 1:
                quark_jets.append(feats)
            else:
                gluon_jets.append(feats)
            break  # one jet per event, matching the single hard jet setup

        if n_try % 2000 == 0:
            print(f"tried {n_try} events -> quark {len(quark_jets)}, gluon {len(gluon_jets)}")

    pythia.stat()

    np.savez(
        "quark_gluon_jets.npz",
        quark_jets=np.array(quark_jets, dtype=object),
        gluon_jets=np.array(gluon_jets, dtype=object),
    )
    print(f"Saved {len(quark_jets)} quark jets and {len(gluon_jets)} gluon jets "
          f"-> quark_gluon_jets.npz  (each jet: array of (z, eta, phi) per particle)")

if __name__ == "__main__":
    main()
