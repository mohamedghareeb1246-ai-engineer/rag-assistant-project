import os
import tempfile
import streamlit as st
from PIL import Image
import chromadb
from ultralytics import YOLO
from groq import Groq

# --- 1. إعدادات الصفحة والـ CSS الاحترافي ---
st.set_page_config(
    page_title="Embedded AI Assistant | Vision & RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تحسين مظهر الواجهة عبر CSS
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #00adb5;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #008c9e;
        border-color: #008c9e;
    }
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00adb5;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. Sidebar - الإعدادات والمعلومات ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/circuit.png", width=80)
    st.title("⚡ التحكم بالنظام")
    st.markdown("---")

    # خيار إدخال المفتاح من الواجهة أو القراءة من المتغيرات
    api_key_input = st.text_input(
        "Groq API Key:",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        help="أدخل مفتاح Groq الخاص بك أو دعه يقرأ المفتاح الافتراضي"
    )

    conf_threshold = st.slider("مستوى ثقة نموذج Vision (Confidence):", 0.10, 0.90, 0.35, 0.05)
    st.markdown("---")
    st.info(
        "💡 **خيارات الاستخدام:**\n- صورة + سؤال نصي\n- صورة فقط (كشف وتوثيق تلقائي)\n- سؤال نصي فقط (استرجاع من الـ RAG)")


# --- 3. تحميل النماذج وقاعدة البيانات (Caching) ---
@st.cache_resource
def load_yolo_model():
    model_path = "notebooks/yolo_trained_model/best.pt"
    return YOLO(model_path)


@st.cache_resource
def load_chroma_client():
    chroma_path = "backend/data/vector_store"
    client = chromadb.PersistentClient(path=chroma_path)
    collections = client.list_collections()
    collection_name = collections[0].name if collections else "langchain"
    return client.get_collection(name=collection_name)


yolo_model = load_yolo_model()
chroma_collection = load_chroma_client()

# --- 4. الهيدر الرئيسي ---
st.title("🧩 Smart Embedded Systems Assistant")
st.caption(
    "نظام ذكي يعتمد على الرؤية الحاسوبية (YOLOv8) وقواعد البيانات المتجهة (RAG) للإجابة التقنية عن المكونات الإلكترونية.")
st.markdown("---")

# --- 5. واجهة المدخلات المقسمة ---
col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.subheader("📷 1. مدخلات الصورة")
    uploaded_file = st.file_uploader("اختر صورة المكون الإلكتروني:", type=["jpg", "jpeg", "png"])

with col_in2:
    st.subheader("💬 2. مدخلات النص")
    user_query = st.text_area("أدخل سؤالك أو استفسارك التقني:",
                              placeholder="مثال: كيف أربط هذا العنصر مع Arduino وما هي أطراف الـ Pinout الخاص به؟",
                              height=130)

submit_btn = st.button("🚀 تشغيل التحليل والاسترجاع", type="primary")

# --- 6. معالجة المدخلات عند الضغط ---
if submit_btn:
    # التحقق من وجود مدخل واحد على الأقل
    if uploaded_file is None and not user_query.strip():
        st.error("❌ يرجى رفع صورة أو كتابة سؤال نصي على الأقل للمتابعة!")
    else:
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 1.2], gap="large")

        query_components = []
        annotated_img = None

        # أ. معالجة الصورة في حال تم رفعها
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_image_path = tmp_file.name

            with st.spinner("🔍 جاري تحليل الصورة بواسطة YOLOv8..."):
                results = yolo_model.predict(source=temp_image_path, conf=conf_threshold, device="cpu")

                # رسم المربعات المكتشفة على الصورة
                res_plotted = results[0].plot()
                annotated_img = Image.fromarray(res_plotted[:, :, ::-1])  # تحويل BGR إلى RGB

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = yolo_model.names[class_id]
                        query_components.append(class_name)

                query_components = list(set(query_components))
                os.remove(temp_image_path)

        # عرض نتيجة الصورة
        with res_col1:
            if uploaded_file is not None:
                st.subheader("🎯 نتيجة الكشف (Vision Output)")
                st.image(annotated_img, caption="المكونات المكتشفة بواسطة YOLO", use_container_width=True)

                if query_components:
                    st.success(f"**المكونات المحددة:** {', '.join(query_components)}")
                else:
                    st.warning("⚠️ لم يتم كشف مكونات محددة فوق نسبة الثقة المختارة.")
            else:
                st.info("ℹ️ تم التشغيل بالوضع النصي فقط (No Image Input).")

        # ب. معالجة الـ RAG والـ LLM
        with res_col2:
            st.subheader("📚 نتائج RAG والرد الذكي")

            # 1. استرجاع السياق من ChromaDB
            retrieved_context = []
            search_targets = query_components if query_components else [user_query]

            with st.spinner("📚 جاري جلب الوثائق التقنية من Vector Store..."):
                for target in search_targets:
                    res = chroma_collection.query(query_texts=[target], n_results=2)
                    if res and 'documents' in res and res['documents']:
                        for doc_list in res['documents']:
                            retrieved_context.extend(doc_list)

                context_text = "\n\n--- Document Excerpt ---\n".join(
                    retrieved_context) if retrieved_context else "No specific context retrieved."

            # 2. بناء الـ Prompt بناءً على نوع المدخل المتاح
            components_str = ", ".join(query_components) if query_components else "None detected from image"
            prompt_query = user_query if user_query.strip() else "Provide full technical specifications, pinout, and circuit connection guidelines for the detected components."

            final_prompt = f"""
You are an expert embedded systems engineer.
Detected Electronic Components: {components_str}

User Request: {prompt_query}

--- Technical Context from Vector Database ---
{context_text}

Provide a clear, highly technical, and structured response including pinouts, operating parameters, and practical application rules where applicable.
"""

            # 3. استدعاء Groq API
            groq_key = api_key_input if api_key_input else "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"

            with st.spinner("⚡ جاري تحليل البيانات وتوليد الإجابة عبر Groq..."):
                try:
                    groq_client = Groq(api_key=groq_key)

                    response = groq_client.chat.completions.create(
                        model="openai/gpt-oss-20b",  # النموذج الرسمي الأحدث المستقر على Groq
                        messages=[{"role": "user", "content": final_prompt}],
                        temperature=0.2
                    )

                    answer = response.choices[0].message.content

                    # عرض الإجابة داخل container أنيق
                    st.markdown("### 💡 التحليل والتعليمات التقنية:")
                    st.markdown(answer)

                    with st.expander("🔍 عرض الـ Context والموجهات الممررة (Prompt Debug)"):
                        st.code(final_prompt, language="markdown")

                except Exception as e:
                    st.error(f"❌ خطأ أثناء استدعاء Groq API: {e}")