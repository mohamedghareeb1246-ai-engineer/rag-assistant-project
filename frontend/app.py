import os
import tempfile

import streamlit as st
from PIL import Image
import chromadb
from ultralytics import YOLO
from groq import Groq

# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Embedded Systems Vision & Retrieval Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_STYLE = """
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    width: 100%;
    border-radius: 6px;
    height: 3em;
    background-color: #2b6cb0;
    color: white;
    font-weight: 600;
    border: none;
    transition: background-color 0.2s ease;
}
.stButton>button:hover {
    background-color: #24578f;
}
.metric-card {
    background-color: #1e2530;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #2b6cb0;
    margin-bottom: 10px;
}
</style>
"""
st.markdown(APP_STYLE, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
YOLO_MODEL_PATH = "notebooks/yolo_trained_model/best.pt"
CHROMA_DB_PATH = "backend/data/vector_store"
GROQ_MODEL_NAME = "openai/gpt-oss-20b"

# -----------------------------------------------------------------------------
# Sidebar - configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Configuration")
    st.markdown("---")

    api_key_input = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="Provide your own Groq API key, or set the GROQ_API_KEY environment variable.",
    )

    conf_threshold = st.slider(
        "Vision model confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    st.markdown("---")
    st.markdown(
        "**Supported input modes**\n"
        "- Image and text query\n"
        "- Image only (automatic component detection and documentation)\n"
        "- Text query only (retrieval from the knowledge base)"
    )

# -----------------------------------------------------------------------------
# Cached resource loaders
# -----------------------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO(YOLO_MODEL_PATH)


@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collections = client.list_collections()
    if not collections:
        raise RuntimeError(
            f"No collections found in the vector store at '{CHROMA_DB_PATH}'."
        )
    return client.get_collection(name=collections[0].name)


def get_groq_client(api_key: str) -> Groq:
    if not api_key:
        raise ValueError(
            "No Groq API key was provided. Set it in the sidebar or via the "
            "GROQ_API_KEY environment variable."
        )
    return Groq(api_key=api_key)


yolo_model = load_yolo_model()
chroma_collection = load_chroma_collection()

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("Smart Embedded Systems Assistant")
st.caption(
    "A technical assistant for electronic components, combining computer vision "
    "(YOLOv8) with retrieval-augmented generation over a technical knowledge base."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# Input section
# -----------------------------------------------------------------------------
input_col_1, input_col_2 = st.columns([1, 1], gap="large")

with input_col_1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Select an image of the electronic component:",
        type=["jpg", "jpeg", "png"],
    )

with input_col_2:
    st.subheader("Text Query")
    user_query = st.text_area(
        "Enter your technical question:",
        placeholder=(
            "Example: How do I connect this component to an Arduino, and what "
            "is its pinout?"
        ),
        height=130,
    )

submit_btn = st.button("Run Analysis", type="primary")

# -----------------------------------------------------------------------------
# Core helper functions
# -----------------------------------------------------------------------------
def run_vision_detection(image_file, confidence: float):
    """Run YOLO detection on the uploaded image and return the annotated image
    and the list of detected component classes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(image_file.getvalue())
        temp_image_path = tmp_file.name

    try:
        results = yolo_model.predict(
            source=temp_image_path, conf=confidence, device="cpu"
        )
        annotated = Image.fromarray(results[0].plot()[:, :, ::-1])  # BGR -> RGB

        detected = set()
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                detected.add(yolo_model.names[class_id])

        return annotated, sorted(detected)
    finally:
        os.remove(temp_image_path)


def retrieve_context(search_terms, n_results: int = 2) -> str:
    """Query the vector store for each search term and concatenate the
    retrieved document excerpts into a single context string."""
    excerpts = []
    for term in search_terms:
        result = chroma_collection.query(query_texts=[term], n_results=n_results)
        documents = result.get("documents") if result else None
        if documents:
            for doc_list in documents:
                excerpts.extend(doc_list)

    if not excerpts:
        return "No specific context retrieved."
    return "\n\n--- Document Excerpt ---\n".join(excerpts)


def build_prompt(components: list[str], query_text: str, context: str) -> str:
    components_str = ", ".join(components) if components else "None detected from image"
    query_str = query_text.strip() or (
        "Provide full technical specifications, pinout, and circuit connection "
        "guidelines for the detected components."
    )

    return f"""You are an expert embedded systems engineer.
Detected electronic components: {components_str}

User request: {query_str}

--- Technical context from the vector database ---
{context}

Provide a clear, technically precise, and well-structured response, including
pinouts, operating parameters, and practical application guidelines where relevant."""


def generate_answer(prompt: str, api_key: str) -> str:
    client = get_groq_client(api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


# -----------------------------------------------------------------------------
# Main processing flow
# -----------------------------------------------------------------------------
if submit_btn:
    if uploaded_file is None and not user_query.strip():
        st.error("Please upload an image or enter a text query before continuing.")
    else:
        st.markdown("---")
        result_col_1, result_col_2 = st.columns([1, 1.2], gap="large")

        detected_components: list[str] = []
        annotated_image = None

        if uploaded_file is not None:
            with st.spinner("Running image analysis..."):
                try:
                    annotated_image, detected_components = run_vision_detection(
                        uploaded_file, conf_threshold
                    )
                except Exception as exc:
                    st.error(f"Image analysis failed: {exc}")

        with result_col_1:
            if uploaded_file is not None:
                st.subheader("Detection Results")
                if annotated_image is not None:
                    st.image(
                        annotated_image,
                        caption="Components detected by the vision model",
                        use_container_width=True,
                    )
                if detected_components:
                    st.success(f"Detected components: {', '.join(detected_components)}")
                else:
                    st.warning(
                        "No components were detected above the selected confidence threshold."
                    )
            else:
                st.info("Running in text-only mode (no image provided).")

        with result_col_2:
            st.subheader("Retrieval and Generated Response")

            search_targets = detected_components if detected_components else [user_query]

            with st.spinner("Retrieving relevant technical documents..."):
                context_text = retrieve_context(search_targets)

            final_prompt = build_prompt(detected_components, user_query, context_text)

            with st.spinner("Generating response..."):
                try:
                    answer = generate_answer(final_prompt, api_key_input)
                    st.markdown("### Technical Analysis")
                    st.markdown(answer)

                    with st.expander("View prompt and retrieved context (debug)"):
                        st.code(final_prompt, language="markdown")

                except Exception as exc:
                    st.error(f"Request to the language model failed: {exc}")
