# ui/results.py
import streamlit as st

def display_metrics(results):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cancer Probability", f"{results['heatmap_stats']['mean_probability']:.1%}")
    with col2:
        st.metric("Malignant Cells", f"{results['malignant_cell_count']}", delta=f"±{results['malignant_cell_uncertainty']:.0%}")
    with col3:
        st.metric("Patches Analysed", f"{results['num_patches']}")
    with col4:
        st.metric("Nuclei Detected", f"{results['nuclei_stats']['count']}")

def render_results_dashboard(results, original_image):
    st.markdown("---")
    st.markdown("## Analysis Results")

    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "AI Interpretation"])

    with tab1:
        display_metrics(results)
        st.markdown("### Heatmap Overlay")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original", use_column_width=True)
        with col2:
            st.image(results['heatmap_overlay'], caption="Probability Heatmap", use_column_width=True)
            
    with tab2:
        # Move the Grade Bars and Detailed Stats logic here
        pass 

    with tab3:
        # Move the HTML Risk Cards and Explanation logic here
        st.write(results['explanation'])