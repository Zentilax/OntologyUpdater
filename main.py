import base64
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI
from rdflib import Graph, Literal, Namespace, RDF, RDFS, OWL
from rdflib.namespace import XSD

app = FastAPI(title="Ontology Updater API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_URI = "http://www.semanticweb.org/bagindokemas/ontologies/2026/1/untitled-ontology-328#"
ns = Namespace(BASE_URI)


# =============================================================================
# OntologyRecommendations
# =============================================================================

class OntologyRecommendations:
    def __init__(self, ontology_schema):
        self.ontology_schema = ontology_schema
        self.new_classes = defaultdict(lambda: {"evidence": [], "suggested_properties": [], "suggested_relations": [], "confidence": 0.0, "sample_data": []})
        self.new_properties = defaultdict(lambda: {"evidence": [], "suggested_domain": [], "suggested_range": None, "confidence": 0.0, "sample_values": []})
        self.new_object_properties = defaultdict(lambda: {"evidence": [], "suggested_domain": None, "suggested_range": None, "confidence": 0.0, "sample_instances": []})
        self.successfully_mapped_properties = set()
        self.unmapped_columns = set()
        self.rejected_properties = set()

    def _is_similar_property(self, p1, p2):
        a = p1.lower().replace("_", "").replace(" ", "")
        b = p2.lower().replace("_", "").replace(" ", "")
        return a == b or a in b or b in a

    def _property_exists(self, name):
        if name in self.ontology_schema["datatype_properties"]:
            return True
        return any(self._is_similar_property(name, e) for e in self.ontology_schema["datatype_properties"])

    def _class_exists(self, name):
        if name in self.ontology_schema["classes"]:
            return True
        return any(e.lower() == name.lower() for e in self.ontology_schema["classes"])

    def add_class_suggestion(self, class_name, evidence, confidence, sample_data=None):
        if self._class_exists(class_name) or confidence < 0.7:
            return
        self.new_classes[class_name]["evidence"].append(evidence)
        self.new_classes[class_name]["confidence"] = max(self.new_classes[class_name]["confidence"], confidence)
        if sample_data:
            self.new_classes[class_name]["sample_data"].append(sample_data)

    def add_property_suggestion(self, property_name, evidence, domain, confidence, sample_value=None):
        if self._property_exists(property_name) or property_name in self.successfully_mapped_properties or confidence < 0.7:
            return
        self.new_properties[property_name]["evidence"].append(evidence)
        if domain not in self.new_properties[property_name]["suggested_domain"]:
            self.new_properties[property_name]["suggested_domain"].append(domain)
        self.new_properties[property_name]["confidence"] = max(self.new_properties[property_name]["confidence"], confidence)
        if sample_value:
            self.new_properties[property_name]["sample_values"].append(sample_value)

    def add_object_property_suggestion(self, property_name, evidence, domain, range_class, confidence, sample=None):
        if property_name in self.ontology_schema["object_properties"] or confidence < 0.75:
            return
        self.new_object_properties[property_name]["evidence"].append(evidence)
        self.new_object_properties[property_name]["suggested_domain"] = domain
        self.new_object_properties[property_name]["suggested_range"] = range_class
        self.new_object_properties[property_name]["confidence"] = max(self.new_object_properties[property_name]["confidence"], confidence)
        if sample:
            self.new_object_properties[property_name]["sample_instances"].append(sample)

    def mark_property_mapped(self, name): self.successfully_mapped_properties.add(name)
    def mark_property_unmapped(self, name): self.unmapped_columns.add(name)
    def mark_property_rejected(self, name): self.rejected_properties.add(name)

    def has_recommendations(self):
        return len(self.new_classes) > 0 or len(self.new_properties) > 0 or len(self.new_object_properties) > 0

    def to_dict(self):
        return {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_recommendations": len(self.new_classes) + len(self.new_properties) + len(self.new_object_properties),
                "new_classes": len(self.new_classes),
                "new_properties": len(self.new_properties),
                "new_object_properties": len(self.new_object_properties),
                "successfully_mapped_properties": len(self.successfully_mapped_properties),
                "truly_unmapped_columns": len(self.unmapped_columns),
            },
            "new_classes": dict(self.new_classes),
            "new_properties": dict(self.new_properties),
            "new_object_properties": dict(self.new_object_properties),
            "debug_info": {
                "successfully_mapped": list(self.successfully_mapped_properties),
                "truly_unmapped": list(self.unmapped_columns),
                "rejected_in_strict_mode": list(self.rejected_properties),
            },
        }


# =============================================================================
# Pure helper functions
# =============================================================================

def sanitize_uri(name):
    if not name or name == "" or (isinstance(name, float) and pd.isna(name)):
        return None
    name = re.sub(r"\s+", "_", str(name))
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    return name or None


def individual_exists(graph, uri):
    return (uri, RDF.type, None) in graph


def validate_property_value(value):
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)) or str(value).lower() in ["nan", "null", "none"]:
        return None
    return value


def fix_owl_namespace_str(content: str) -> str:
    if "xmlns:owl=" in content:
        return content
    return content.replace("<rdf:RDF\n", '<rdf:RDF\n   xmlns:owl="http://www.w3.org/2002/07/owl#"\n', 1)


def extract_ontology_schema(graph):
    schema = {"classes": [], "datatype_properties": [], "object_properties": []}
    for s in graph.subjects(RDF.type, OWL.Class):
        name = str(s).split("#")[-1]
        if name and not name.startswith("_"):
            schema["classes"].append(name)
    for s in graph.subjects(RDF.type, OWL.DatatypeProperty):
        name = str(s).split("#")[-1]
        if name and not name.startswith("_"):
            schema["datatype_properties"].append(name)
    for s in graph.subjects(RDF.type, OWL.ObjectProperty):
        name = str(s).split("#")[-1]
        if name and not name.startswith("_"):
            schema["object_properties"].append(name)
    return schema


def auto_detect_cabai_format(df):
    first_col = str(df.columns[0]).lower().strip()
    second_col = str(df.columns[1]).lower().strip()
    if any(k in first_col for k in ["nomor", "no", "number", "id"]):
        return True
    if any(k in second_col for k in ["karakter", "character", "property", "atribut"]):
        return True
    if df.shape[1] > df.shape[0] * 1.5:
        return True
    return False


def untranspose_cabai_csv(df_transposed):
    first_col = df_transposed.columns[0]
    second_col = df_transposed.columns[1]
    df_work = df_transposed.drop(columns=[first_col])
    df_work = df_work.set_index(second_col)
    df_untransposed = df_work.T.reset_index()
    df_untransposed = df_untransposed.rename(columns={"index": "NamaCabai"})
    df_untransposed.columns = df_untransposed.columns.str.strip()
    df_untransposed = df_untransposed.replace("", pd.NA)
    return df_untransposed


def map_row_with_llm(client, row_dict, ontology_schema):
    filtered_dict = {k: v for k, v in row_dict.items() if not (isinstance(v, float) and pd.isna(v)) and v != ""}
    if not filtered_dict:
        return None

    prompt = f"""You are mapping CSV data into an OWL ontology.

Classes: {', '.join(ontology_schema['classes'])}
Datatype Properties: {', '.join(ontology_schema['datatype_properties']) if ontology_schema['datatype_properties'] else 'None'}
Object Properties: {', '.join(ontology_schema['object_properties']) if ontology_schema['object_properties'] else 'berasalDari'}

CSV DATA:
{json.dumps(filtered_dict, indent=2, ensure_ascii=False)}

RULES:
1. Identify cabai/entity name → "cabai_name"
2. Identify region/location → "daerah_asal"
3. Map ALL other columns to existing properties by SEMANTIC MEANING (fuzzy match)
4. Only put in "unmapped_properties" if truly cannot map to any existing property

OUTPUT (strict JSON):
{{
  "cabai_name": "Gendot",
  "daerah_asal": "Lombok",
  "datatype_properties": {{"TinggiTanaman": "100 cm"}},
  "unmapped_properties": {{}}
}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Expert ontology mapper. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        result = json.loads(response.choices[0].message.content)
        return result if "cabai_name" in result else None
    except Exception as e:
        return None


def analyze_recommendations(client, df, ontology_schema, recommendations):
    if not recommendations.unmapped_columns:
        return {"new_classes": [], "new_datatype_properties": [], "new_object_properties": []}

    cols = [c for c in recommendations.unmapped_columns if c in df.columns]
    if not cols:
        return {"new_classes": [], "new_datatype_properties": [], "new_object_properties": []}

    sample_data = df[cols].head(5).to_dict("records")

    prompt = f"""You are an ontology expert analyzing ONLY truly unmapped columns.

CURRENT ONTOLOGY:
Classes: {', '.join(ontology_schema['classes'])}
Datatype Properties: {', '.join(ontology_schema['datatype_properties'])}
Object Properties: {', '.join(ontology_schema['object_properties'])}

UNMAPPED COLUMNS: {', '.join(cols)}
SAMPLE DATA: {json.dumps(sample_data, indent=2, ensure_ascii=False)}

RULES: Be VERY conservative. Only recommend if confidence >= 0.75. Return JSON only.

{{"new_classes": [], "new_datatype_properties": [], "new_object_properties": []}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Conservative ontology expert. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {"new_classes": [], "new_datatype_properties": [], "new_object_properties": []}


# =============================================================================
# Main processing function — uses a log list instead of print()
# =============================================================================

def process_ontology(csv_bytes: bytes, owl_bytes: bytes, enable_recommendations: bool):
    logs = []

    def log(msg: str):
        print(msg)          # still prints to Railway console
        logs.append(msg)    # also captured for frontend

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set on server")

    client = OpenAI(api_key=api_key)

    # ── Load CSV
    log("=" * 60)
    log("ONTOLOGY UPDATE PROCESS")
    log("=" * 60)

    df_raw = pd.read_csv(io.BytesIO(csv_bytes))
    log(f"\n✓ CSV loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")

    if auto_detect_cabai_format(df_raw):
        log("  Transposed format detected — untransposing...")
        df = untranspose_cabai_csv(df_raw)
        log(f"  After untranspose: {df.shape[0]} rows × {df.shape[1]} columns")
    else:
        df = df_raw
        log("  Standard row-per-entity format")

    # ── Load ontologyhh
    log("\n🔍 Loading ontology...")
    g = Graph()
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.parse(data=owl_bytes, format="xml")
    triple_count = len(g)

    ontology_schema = extract_ontology_schema(g)
    log(f"✓ Classes ({len(ontology_schema['classes'])}): {', '.join(ontology_schema['classes'])}")
    log(f"✓ Datatype properties: {len(ontology_schema['datatype_properties'])}")
    log(f"✓ Object properties: {len(ontology_schema['object_properties'])}")

    recommendations = OntologyRecommendations(ontology_schema)

    stats = {
        "total_rows": len(df),
        "successful": 0,
        "failed": 0,
        "new_cabai": 0,
        "new_daerah": 0,
        "skipped": 0,
        "existing_cabai_updated": 0,
        "total_properties_added": 0,
        "new_properties_created": 0,
        "properties_rejected": 0,
    }

    total = len(df)
    log(f"\n🔄 Processing {total} rows...")
    log("-" * 60)

    for idx, row in df.iterrows():
        row_num = idx + 1
        try:
            row_dict = row.to_dict()
            log(f"\n[Row {row_num}/{total}] Calling GPT to map row...")

            mapping = map_row_with_llm(client, row_dict, ontology_schema)

            if mapping is None:
                log(f"  ⚠ Skipped — LLM returned no mapping")
                stats["skipped"] += 1
                continue

            cabai_name = sanitize_uri(mapping.get("cabai_name"))
            daerah_name = sanitize_uri(mapping.get("daerah_asal"))

            if not cabai_name:
                log(f"  ⚠ Skipped — no cabai name identified")
                stats["skipped"] += 1
                continue

            log(f"  ✓ Cabai: {cabai_name}" + (f"  |  Daerah: {daerah_name}" if daerah_name else ""))

            cabai_uri = ns[cabai_name]
            if not individual_exists(g, cabai_uri):
                g.add((cabai_uri, RDF.type, OWL.NamedIndividual))
                g.add((cabai_uri, RDF.type, ns.Cabai))
                stats["new_cabai"] += 1
                log(f"  + New Cabai individual created")
            else:
                if (cabai_uri, RDF.type, OWL.NamedIndividual) not in g:
                    g.add((cabai_uri, RDF.type, OWL.NamedIndividual))
                stats["existing_cabai_updated"] += 1
                log(f"  ~ Updating existing Cabai")

            if daerah_name:
                daerah_uri = ns[daerah_name]
                if not individual_exists(g, daerah_uri):
                    g.add((daerah_uri, RDF.type, OWL.NamedIndividual))
                    g.add((daerah_uri, RDF.type, ns.DaerahAsal))
                    stats["new_daerah"] += 1
                    log(f"  + New DaerahAsal: {daerah_name}")
                else:
                    if (daerah_uri, RDF.type, OWL.NamedIndividual) not in g:
                        g.add((daerah_uri, RDF.type, OWL.NamedIndividual))
                if (cabai_uri, ns.berasalDari, daerah_uri) not in g:
                    g.add((cabai_uri, ns.berasalDari, daerah_uri))

            properties_added = 0
            for prop, value in mapping.get("datatype_properties", {}).items():
                clean_value = validate_property_value(value)
                if clean_value is None:
                    continue

                in_schema = prop in ontology_schema["datatype_properties"]
                if not in_schema:
                    stats["new_properties_created"] += 1
                    log(f"  ⚡ New property created: {prop}")

                recommendations.mark_property_mapped(prop)
                prop_uri = ns[prop]

                if isinstance(clean_value, bool):
                    literal = Literal(clean_value, datatype=XSD.boolean)
                elif isinstance(clean_value, int):
                    literal = Literal(clean_value, datatype=XSD.integer)
                elif isinstance(clean_value, float):
                    literal = Literal(clean_value, datatype=XSD.float)
                else:
                    literal = Literal(str(clean_value), datatype=XSD.string)

                g.remove((cabai_uri, prop_uri, None))
                g.add((cabai_uri, prop_uri, literal))
                properties_added += 1

            for unmapped_col in mapping.get("unmapped_properties", {}).keys():
                recommendations.mark_property_unmapped(unmapped_col)
                log(f"  ? Unmapped column: {unmapped_col}")

            stats["total_properties_added"] += properties_added
            stats["successful"] += 1
            log(f"  ✓ Done — {properties_added} properties added")

        except Exception as e:
            log(f"  ✗ Error on row {row_num}: {str(e)}")
            stats["failed"] += 1

    # ── Serialize
    log("\n💾 Serializing ontology...")
    raw = g.serialize(format="xml")
    owl_str = raw if isinstance(raw, str) else raw.decode("utf-8")
    owl_str = fix_owl_namespace_str(owl_str)
    stats["triples_added"] = len(g) - triple_count
    log(f"✓ Serialized — {stats['triples_added']} triples added")

    # ── Recommendations
    rec_dict = {}
    if enable_recommendations:
        log("\n🔍 Analysing unmapped columns for recommendations...")
        analysis = analyze_recommendations(client, df, ontology_schema, recommendations)
        for cs in analysis.get("new_classes", []):
            if cs.get("confidence", 0) >= 0.80:
                recommendations.add_class_suggestion(cs["name"], cs["evidence"], cs["confidence"], cs.get("suggested_properties"))
        for ps in analysis.get("new_datatype_properties", []):
            if ps.get("confidence", 0) >= 0.75:
                for domain in ps.get("suggested_domain", ["Unknown"]):
                    recommendations.add_property_suggestion(ps["name"], ps["evidence"], domain, ps["confidence"], ps.get("sample_values", [None])[0])
        for op in analysis.get("new_object_properties", []):
            if op.get("confidence", 0) >= 0.85:
                recommendations.add_object_property_suggestion(op["name"], op["evidence"], op["domain"], op["range"], op["confidence"])
        rec_dict = recommendations.to_dict()
        n = rec_dict["summary"]["total_recommendations"]
        log(f"✓ {n} recommendation(s) generated")
    else:
        log("\nRecommendations skipped (disabled by user)")
        rec_dict = recommendations.to_dict()

    # ── Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Total rows:           {stats['total_rows']}")
    log(f"Successfully processed: {stats['successful']}")
    log(f"Skipped:              {stats['skipped']}")
    log(f"Failed:               {stats['failed']}")
    log(f"New Cabai instances:  {stats['new_cabai']}")
    log(f"Properties added:     {stats['total_properties_added']}")
    log(f"New properties:       {stats['new_properties_created']}")
    log(f"Triples added:        {stats['triples_added']}")
    log("=" * 60)

    return owl_str, stats, rec_dict, logs


# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "Ontology Updater API v3"}


@app.get("/health")
def health():
    return {"status": "healthy", "openai_key_set": bool(os.environ.get("OPENAI_API_KEY"))}


@app.post("/update-ontology")
async def update_ontology(
    csv_file: UploadFile = File(...),
    owl_file: UploadFile = File(...),
    enable_recommendations: bool = Form(True),
):
    if not csv_file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="csv_file must be a .csv file")
    if not owl_file.filename.lower().endswith(".owl"):
        raise HTTPException(status_code=400, detail="owl_file must be a .owl file")
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    csv_bytes = await csv_file.read()
    owl_bytes = await owl_file.read()

    try:
        owl_str, stats, recommendations, logs = process_ontology(
            csv_bytes=csv_bytes,
            owl_bytes=owl_bytes,
            enable_recommendations=enable_recommendations,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    owl_b64 = base64.b64encode(owl_str.encode("utf-8")).decode("ascii")
    owl_filename = owl_file.filename.replace(".owl", "_updated.owl")

    return JSONResponse({
        "success": True,
        "stats": stats,
        "recommendations": recommendations,
        "logs": logs,
        "owl_file_b64": owl_b64,
        "owl_filename": owl_filename,
    })
