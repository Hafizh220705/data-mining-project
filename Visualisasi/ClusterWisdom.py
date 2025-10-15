# Visualisasi/ClusterWisdom.py
# Menyusun rencana aksi (Wisdom layer) dari insight cluster.
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

# --------------------- KPI & Template tindakan ---------------------

_KPI_TEMPLATES = {
    "Education": [
        ("IPK rata-rata", "≥ +0.10 dari baseline per semester"),
        ("Tingkat kelulusan mata kuliah inti", "≥ +5 pp dalam 1 tahun"),
        ("Kepuasan pengajaran (Likert)", "≥ +0.3 poin dalam 1 semester"),
    ],
    "Financial": [
        ("Mahasiswa dengan tekanan finansial tinggi", "≤ -20% dalam 1 semester"),
        ("Partisipasi literasi finansial", "≥ 60% target cluster dalam 1 semester"),
        ("Keterlambatan pembayaran UKT", "≤ -15% dalam 1 tahun"),
    ],
    "Physical": [
        ("Jam tidur (self-report)", "≥ +0.5 jam/hari dalam 2 bulan"),
        ("Frekuensi olahraga/minggu", "≥ +1 sesi dalam 2 bulan"),
        ("Kunjungan layanan kesehatan kampus (preventif)", "≥ +10% dalam 1 semester"),
    ],
    "Psychological": [
        ("Skor distress (self-report)", "≤ -0.4 z dalam 3 bulan"),
        ("Penggunaan layanan konseling", "≥ +25% dari baseline (yang butuh)"),
        ("Kehadiran workshop coping-stress", "≥ 50% target cluster"),
    ],
    "Relational": [
        ("Keterlibatan komunitas/UKM", "≥ +15% dalam 1 semester"),
        ("Skor dukungan sosial", "≥ +0.3 poin dalam 1 semester"),
        ("Retensi mahasiswa berisiko", "≥ +5 pp dalam 1 tahun"),
    ],
}

_ACTION_TEMPLATES = {
    "Education": {
        "pos": [
            "Skalakan program mentoring/peer-tutoring yang terbukti efektif.",
            "Standardisasi materi & rubrik penilaian antarmata kuliah unggulan."
        ],
        "neg": [
            "Klinik belajar mingguan untuk mata kuliah inti dengan tutor senior.",
            "Audit beban tugas & penyelarasan LO antar-dosen."
        ]
    },
    "Financial": {
        "pos": [
            "Kurikulum literasi finansial lanjutan (budgeting proaktif, investasi dasar).",
            "Optimasi pencocokan beasiswa dan kerja paruh waktu on-campus."
        ],
        "neg": [
            "Dana darurat & opsi cicilan UKT fleksibel; sosialisasi jalur pengajuan cepat.",
            "Kemitraan kerja paruh waktu dengan jadwal ramah akademik."
        ]
    },
    "Physical": {
        "pos": [
            "Kampanye sleep hygiene & akses fasilitas olahraga yang konsisten.",
            "Paket nutrisi sehat untuk mahasiswa aktif organisasi."
        ],
        "neg": [
            "Program habit-building aktivitas fisik ringan terstruktur.",
            "Skrining kesehatan berkala & rujukan cepat."
        ]
    },
    "Psychological": {
        "pos": [
            "Workshop coping-stress & mindfulness tingkat lanjut.",
            "Pelatihan kader sebaya (gatekeeper)."
        ],
        "neg": [
            "Perluas kapasitas konseling; SLA & alur rujukan jelas.",
            "Kelas manajemen stres berbasis CBT singkat (4–6 sesi)."
        ]
    },
    "Relational": {
        "pos": [
            "Fasilitasi komunitas minat & proyek kolaboratif lintas jurusan.",
            "Program buddy untuk mahasiswa baru/transfer."
        ],
        "neg": [
            "Kelompok dukungan sosial terjadwal (10–12 orang).",
            "Pelatihan komunikasi asertif & resolusi konflik."
        ]
    }
}

# --------------------- Util skoring & format ---------------------

def _priority_score(value: float, n_members: int, basis: str = "Z") -> float:
    """Prioritas = deviasi * sqrt(n). Basis Z: |z|; basis mean: |value-3|."""
    dev = abs(float(value)) if basis.upper().startswith("Z") else abs(float(value) - 3.0)
    return dev * np.sqrt(max(1, n_members))

def _cost_band(priority: float) -> str:
    """
    Estimasi pita biaya berdasarkan prioritas (heuristik sederhana).
    Kamu bebas ganti rule of thumb ini sesuai kampusmu.
    """
    if priority >= 12:
        return "Tinggi"
    if priority >= 6:
        return "Sedang"
    return "Rendah"

def _owners_default(aspect: str, is_positive: bool) -> List[str]:
    """Saran owner default per aspek."""
    mapping = {
        "Education": ["Wakil Dekan Akademik", "Koordinator Mata Kuliah"],
        "Financial": ["Biro Keuangan", "Unit Beasiswa"],
        "Physical": ["Unit Kesehatan Kampus", "Kemahasiswaan"],
        "Psychological": ["Layanan Konseling", "Kemahasiswaan"],
        "Relational": ["BEM/UKM", "Kemahasiswaan"],
    }
    return mapping.get(aspect, ["Kemahasiswaan"])

def _timeline_default(is_positive: bool) -> Tuple[str, str]:
    """Saran timeline default (start, due) — bisa kamu override di UI."""
    return ("Bulan 1", "Bulan 3") if is_positive else ("Bulan 1", "Bulan 2")

# --------------------- API utama ---------------------

def build_action_plan(
    recommendations: List[Dict[str, object]],
    basis_name: str = "Z-score",
    style: str = "formal",
    override_owners: Optional[Dict[str, List[str]]] = None,
    override_timeline: Optional[Tuple[str, str]] = None,
) -> pd.DataFrame:
    """
    Ubah output make_policy_recommendations() → DataFrame rencana aksi.
    Kolom: cluster, n, aspect, direction, score, priority, action, owners, start, due, kpi_name, kpi_target, cost_band
    """
    rows = []
    is_z = basis_name.upper().startswith("Z")

    for r in recommendations:
        cid = str(r["cluster"])
        n = int(r["n"])
        for direction, key in [("positives", "pos"), ("concerns", "neg")]:
            items = r.get(direction, [])
            for it in items:
                aspect = str(it["aspect"])
                score = float(it["score"])
                prio = float(it["priority"])
                actions = list(it.get("actions", [])) or _ACTION_TEMPLATES.get(aspect, {}).get("pos" if direction=="positives" else "neg", ["(aksi belum ditentukan)"])
                owners = (override_owners or {}).get(aspect, _owners_default(aspect, direction=="positives"))
                start, due = override_timeline if override_timeline else _timeline_default(direction=="positives")
                # KPI
                kpis = _KPI_TEMPLATES.get(aspect, [("KPI khusus aspek", "tetapkan target")])
                kpi_name, kpi_target = kpis[0]

                for act in actions:
                    rows.append({
                        "cluster": cid,
                        "n": n,
                        "aspect": aspect,
                        "direction": "Kekuatan" if direction=="positives" else "Isu",
                        "score": score,
                        "priority": prio,
                        "action": act,
                        "owners": ", ".join(owners),
                        "start": start,
                        "due": due,
                        "kpi_name": kpi_name,
                        "kpi_target": kpi_target,
                        "cost_band": _cost_band(prio),
                        "basis": basis_name,
                    })

    df_plan = pd.DataFrame(rows)
    if not df_plan.empty:
        # urutkan: cluster → Isu dulu (umumnya prioritas) → priority desc
        df_plan["direction_order"] = df_plan["direction"].map({"Isu": 0, "Kekuatan": 1})
        df_plan = df_plan.sort_values(["cluster", "direction_order", "priority"], ascending=[True, True, False]).drop(columns=["direction_order"])
    return df_plan

def plan_to_markdown(df_plan: pd.DataFrame, pretty_names: Optional[Dict[str, str]] = None) -> str:
    """Konversi rencana aksi → Markdown (siap diunduh)."""
    if df_plan is None or df_plan.empty:
        return "# Rencana Aksi\n\n(Tidak ada item.)\n"

    lines = ["# Rencana Aksi per Cluster", ""]
    for cid, sub in df_plan.groupby("cluster"):
        lines.append(f"## Cluster {cid} — Anggota: {int(sub['n'].max())}")
        for _, r in sub.iterrows():
            nm = (pretty_names or {}).get(r["aspect"], r["aspect"])
            lines.append(f"- **{r['direction']} — {nm}** (skor: {r['score']:.2f}, prioritas: {r['priority']:.2f}, biaya: {r['cost_band']})")
            lines.append(f"  - Aksi: {r['action']}")
            lines.append(f"  - Penanggung jawab: {r['owners']} | Waktu: {r['start']} → {r['due']}")
            lines.append(f"  - KPI: {r['kpi_name']} — Target: {r['kpi_target']}")
        lines.append("")  # spasi
    return "\n".join(lines)

# --------------------- Demo mandiri ---------------------
if __name__ == "__main__":
    # Contoh minimal dengan struktur rekomendasi palsu
    fake_recs = [
        {"cluster": "0", "n": 120,
         "positives": [{"aspect": "Education", "score": 0.8, "priority": 10, "actions": []}],
         "concerns":  [{"aspect": "Psychological", "score": -0.6, "priority": 12, "actions": []}]},
    ]
    plan = build_action_plan(fake_recs, basis_name="Z-score")
    print(plan.head())
    print(plan_to_markdown(plan))
