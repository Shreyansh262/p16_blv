"""
02_build_rag_db.py
Builds SQLite RAG database with precomputed scene embeddings.
Usage:
    python 02_build_rag_db.py
"""

import json, sqlite3, logging
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

SCENE_LIBRARY = {
    "outdoor_street": (
        "Urban street environment. Pavement beneath you with road traffic "
        "to one or both sides. A raised curb separates the sidewalk from the "
        "road. Pedestrians may be present. Street furniture such as poles, "
        "bins, and benches may line the path."
    ),
    "outdoor_crosswalk": (
        "Pedestrian crossing area. Road surface directly ahead with traffic "
        "signals visible overhead or to the side. Tactile paving strips mark "
        "the crossing boundary beneath your feet. Vehicles may be present "
        "on either side."
    ),
    "outdoor_park": (
        "Open outdoor green space. Grass or paved paths extend ahead. "
        "Benches and low obstacles such as tree roots or uneven surfaces "
        "may be present. Other people and dogs may be moving through the area."
    ),
    "outdoor_parking_lot": (
        "Open parking area. Vehicle bays extend ahead and to the sides. "
        "Moving vehicles are possible. Painted lines mark bays on the ground. "
        "Speed bumps may be present. Surface is typically level tarmac."
    ),
    "outdoor_bus_stop": (
        "Public transport waiting area. Shelter structure visible to one side. "
        "Pavement edge and road directly adjacent. Other waiting passengers "
        "may be seated or standing nearby."
    ),
    "outdoor_stairs": (
        "Outdoor stairs or escalator. Steps descend or ascend directly ahead. "
        "Handrails present on one or both sides. Step edges may have tactile "
        "or colored markings."
    ),
    "outdoor_construction": (
        "Construction or roadworks area. Temporary barriers redirect the "
        "footpath. Uneven or loose surface beneath feet. High hazard zone. "
        "Alternative path marked by temporary signage."
    ),
    "indoor_corridor": (
        "Corridor or hallway extending ahead. Walls close on both sides. "
        "Doors may open from either side without warning. Floor is typically "
        "smooth and level. Width allows one to two people to pass side by side."
    ),
    "indoor_staircase": (
        "Indoor staircase. Steps ascend or descend directly ahead. "
        "Handrail present on at least one side. Step edges may be marked. "
        "Landing areas at top and bottom."
    ),
    "indoor_elevator": (
        "Elevator lobby area. Elevator doors directly ahead or to the side. "
        "Call buttons mounted on the wall. Gap between elevator floor and "
        "building floor when doors open. Other people may be waiting nearby."
    ),
    "indoor_entrance": (
        "Building entrance or exit. Door or automatic door directly ahead. "
        "Threshold or small step may be present at door base. "
        "Security barriers or turnstiles may be present."
    ),
    "indoor_ramp": (
        "Ramp surface. Floor slopes upward or downward ahead. "
        "Handrails on one or both sides. Level surface at top and bottom."
    ),
    "indoor_kitchen": (
        "Kitchen environment. Counter surfaces extend along walls at "
        "approximately waist height. Appliances including stove, microwave, "
        "and refrigerator positioned along the walls. "
        "Wet floor hazard near sink area."
    ),
    "indoor_living_room": (
        "Living room or lounge area. Sofa and seating furniture occupy the "
        "central space. Coffee table at low height presents trip hazard. "
        "Open space in center of room for movement."
    ),
    "indoor_bedroom": (
        "Bedroom environment. Bed occupies significant floor space against "
        "one wall. Bedside tables at low height on either side. "
        "Clear floor path from door to bed."
    ),
    "indoor_bathroom": (
        "Bathroom environment. Wet floor hazard throughout. Toilet to one "
        "side, sink and mirror ahead or adjacent. Shower or bath area to "
        "one side. Confined space with limited movement area."
    ),
    "indoor_office": (
        "Office or workspace. Desk surfaces at sitting height throughout. "
        "Chair wheels and desk legs present as low obstacles. "
        "Cables may cross floor paths. Narrow aisles between workstations."
    ),
    "indoor_cafeteria": (
        "Cafeteria or dining hall. Food serving counter typically at far end "
        "approximately 8 to 15 meters ahead. Tables and chairs occupy central "
        "floor. Queuing area and trays to the left of serving counter. "
        "Floor may be wet near drink stations."
    ),
    "indoor_hospital": (
        "Medical facility interior. Reception desk or nursing station ahead "
        "or to one side. Staff in scrubs or white coats present. "
        "Wheelchairs and medical equipment may be in corridors. "
        "Smooth linoleum floor."
    ),
    "indoor_shop": (
        "Retail shop interior. Product shelving units form aisles ahead. "
        "Shelves at various heights including low shelves near floor. "
        "Shopping baskets may block aisles. Checkout counters at far end."
    ),
    "indoor_library": (
        "Library environment. Tall bookshelves form narrow aisles. "
        "Reading tables occupy open areas. Low stools and bags on floor "
        "present trip hazards."
    ),
    "indoor_classroom": (
        "Classroom or lecture space. Rows of desks and chairs face a board "
        "at the front. Narrow gaps between desk rows. "
        "Door typically at rear of room."
    ),
    "indoor_gym": (
        "Gym or exercise facility. Large exercise equipment occupies floor "
        "space. Narrow paths between machines. Floor is rubber matting. "
        "Moving equipment presents hazard."
    ),
    "indoor_supermarket": (
        "Supermarket environment. Wide aisles with tall shelving on both "
        "sides. Refrigerated sections along walls. Promotional displays may "
        "protrude into aisles. Wet floor possible near produce section."
    ),
    "indoor_transport_hub": (
        "Transport hub interior such as train station or airport. Moving "
        "crowds with luggage present hazard. Multiple pathways and lanes. "
        "Moving walkways possible. Level smooth floor."
    ),
}


def build_rag_database(scene_library, db_path,
                       model_name="all-MiniLM-L6-v2"):
    log.info(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    descriptions = list(scene_library.values())
    categories   = list(scene_library.keys())

    log.info(f"Encoding {len(descriptions)} scene descriptions...")
    embeddings = embedder.encode(
        descriptions,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("DROP TABLE IF EXISTS scene_library")
    conn.execute("""
        CREATE TABLE scene_library (
            id          INTEGER PRIMARY KEY,
            category    TEXT    NOT NULL,
            description TEXT    NOT NULL,
            embedding   BLOB    NOT NULL,
            emb_dim     INTEGER NOT NULL
        )
    """)

    for i, (cat, desc, emb) in enumerate(
            zip(categories, descriptions, embeddings)):
        conn.execute(
            "INSERT INTO scene_library VALUES (?, ?, ?, ?, ?)",
            (i, cat, desc,
             emb.astype(np.float32).tobytes(), len(emb))
        )
    conn.commit()

    count = conn.execute(
        "SELECT COUNT(*) FROM scene_library"
    ).fetchone()[0]
    db_size_kb = db_path.stat().st_size / 1024
    log.info(f"Database built: {count} scenes | "
             f"Size: {db_size_kb:.1f} KB | Path: {db_path}")
    conn.close()
    return embedder


def query_rag(query_text, db_path, embedder, top_k=1):
    """Use this at training time to retrieve scene context."""
    query_emb = embedder.encode(
        [query_text], normalize_embeddings=True
    )[0]

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT category, description, embedding FROM scene_library"
    ).fetchall()
    conn.close()

    scores = []
    for cat, desc, emb_bytes in rows:
        stored = np.frombuffer(emb_bytes, dtype=np.float32)
        score  = float(np.dot(query_emb, stored))
        scores.append({"category": cat,
                       "description": desc,
                       "score": score})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]


if __name__ == "__main__":
    config_path = (Path(__file__).parent.parent.parent
                   / "config" / "paths_config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    pr      = cfg["project_root"]
    db_path = Path(cfg["data"]["rag_db"])
    lib_out = Path(cfg["data"]["scene_library"])

    lib_out.parent.mkdir(parents=True, exist_ok=True)
    with open(lib_out, "w") as f:
        json.dump(SCENE_LIBRARY, f, indent=2)
    log.info(f"Scene library saved: {lib_out}")

    embedder = build_rag_database(SCENE_LIBRARY, db_path)

    # Quick test
    results = query_rag(
        "person cooking at stove", db_path, embedder
    )
    log.info(f"Test query result → {results[0]['category']}")
    log.info("RAG database ready.")
