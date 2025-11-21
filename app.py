import streamlit as st
from backend import predict, get_model_instance

st.set_page_config(page_title="🧠 Disease Predictor", layout="centered")

st.title("🩺 Symptom-Based Disease Prediction (ML Model)")
st.markdown("Select your symptoms to predict possible diseases.")

# Load model
model = get_model_instance()
symptom_options = model.features or [
    "fever", "cough", "headache", "fatigue", "runny_nose", "sneezing", "rash", "nausea", "vomiting"
]

selected = st.multiselect("Choose your symptoms:", symptom_options)

if st.button("🔍 Predict Disease"):
    if not selected:
        st.warning("Please select at least one symptom.")
    else:
        with st.spinner("Analyzing symptoms..."):
            results = predict(selected)

        st.subheader("Prediction Results")
        for res in results[:3]:
            st.markdown(f"**🧾 Disease:** {res['disease']}")
            st.markdown(f"**📊 Probability:** {res['probability']*100:.1f}%")
            st.markdown(f"**💡 Precautions:** {res['precautions']}")
            st.write("---")

st.caption("Built with ❤️ using Streamlit & scikit-learn")
