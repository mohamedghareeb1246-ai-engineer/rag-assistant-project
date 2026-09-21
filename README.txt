# 🧩 Smart Embedded Systems Assistant (Vision + RAG)


Test my project(https://rag-assistant-project-dypz9ilaybjwqgdwsfpj5u.streamlit.app/).



نظام ذكي متكامل يعتمد على الرؤية الحاسوبية (Computer Vision) وتقنية استرجاع البيانات المعزز بالتوليد (RAG) لتقديم استشارات وتفاصيل تقنية دقيقة حول المكونات الإلكترونية والميكروكنترولر بناءً على الصور والأسئلة النصية.

--------------------------------------------------

## 🌟 مميزات المشروع

* Vision Object Detection: التعرف اللحظي على العناصر الإلكترونية في الصور باستخدام نموذج YOLOv8 المدرب.
* Smart RAG Pipeline: البحث في قواعد البيانات المتجهة (ChromaDB) لاستخراج المعلومات والـ Datasheets الموثوقة من كتب الميكروكنترولر.
* LLM Integration: توليد إجابات هندسية تفاعلية وسريعة للغاية باستخدام Groq API (llama-3.3-70b-versatile).
* Interactive Web GUI: واجهة مستخدم احترافية وتفاعلية باستخدام Streamlit.
* Flexible Input Options: دعم كامل للإدخال متعدد الأنماط:
  - صورة + سؤال نصي
  - صورة فقط (كشف وإجابة تلقائية)
  - سؤال نصي فقط (مساعد شات تقني)

--------------------------------------------------

## 📁 هيكل المشروع (Project Structure)

rag-assistant-project/
│
├── backend/
│   └── data/
│       └── vector_store/       # قاعدة البيانات المتجهة (ChromaDB)
│           ├── chroma.sqlite3
│           └── <uuid_folder>/
│
├── data/
│   ├── documents/             # المصادر والمستندات التقنية (PDFs)
│   │   └── MICROCONTROLLERBOOK.pdf
│   └── images/                # الصور المرفقة لاختبار النظام
│       └── test_component.jpg
│
├── frontend/
│   └── app.py                 # تطبيق واجهة المستخدم (Streamlit App)
│
├── notebooks/                 # بيئة التدريب والتجارب
│   ├── Arduino-5/             # بيانات داتاسيت المكونات
│   ├── runs/detect/weights/   # أوزان نموذج YOLOv8 المدرب (best.pt)
│   ├── rag_pipeline.ipynb     # المفكرة التفاعلية لتجربة الـ Pipeline
│   └── yolov8n.pt             # الموديل الأساسي لـ YOLO
│
├── requirements.txt           # المكتبات المطلوبة للمشروع
└── README.txt                 # توثيق المشروع

--------------------------------------------------

## ⚙️ متطلبات التشغيل (Prerequisites)

* Python: 3.9 أو أحدث.
* Groq API Key: مفتاح مجاني للحصول على استجابة سريعة من الـ LLM عبر Groq Console (https://console.groq.com/).

--------------------------------------------------

## 🚀 كيفية التثبيت والتشغيل (Setup & Run)

1. استنساخ المشروع وتجهيز البيئة الافتراضية:
   python -m venv .venv

   # تفعيل البيئة الافتراضية (Windows):
   .venv\Scripts\activate

   # تفعيل البيئة الافتراضية (Linux/Mac):
   source .venv/bin/activate

2. تثبيت المكتبات المطلوبة:
   pip install streamlit ultralytics chromadb groq pillow

3. إعداد مفتاح البيئة (اختياري):
   # Windows (CMD):
   set GROQ_API_KEY=your_groq_api_key_here

   # Linux/Mac:
   export GROQ_API_KEY="your_groq_api_key_here"

4. تشغيل الواجهة (Streamlit App):
   streamlit run frontend/app.py

--------------------------------------------------

## 🔄 كيف يعمل النظام؟ (System Architecture)

[المستخدم: صورة / نص]
        │
        ▼
{هل يوجد صورة؟} ─── نعم ───► [YOLOv8 Detection] ───► [استخراج أسماء المكونات] ──┐
        │                                                                       │
        لا                                                                      │
        │                                                                       │
        ▼                                                                       ▼
[استخدام السؤال النصي مباشرة] ──────────────────────────────────────────► [ChromaDB Vector Search]
                                                                                │
                                                                                ▼
                                                            [استرجاع السياق من كتاب الميكروكنترولر]
                                                                                │
                                                                                ▼
                                                                     [صياغة الـ Prompt المدمج]
                                                                                │
                                                                                ▼
                                                                     [Groq Llama-3.3-70b]
                                                                                │
                                                                                ▼
                                                            [عرض الرد التقني والـ Pinout في Streamlit]

--------------------------------------------------

## 🛠️ التقنيات المستخدمة (Tech Stack)

* UI/Frontend: Streamlit
* Computer Vision: Ultralytics YOLOv8
* Vector Database: ChromaDB
* LLM Orchestration: Groq Cloud SDK (Llama-3.3-70B)
* Programming Language: Python 3.10
