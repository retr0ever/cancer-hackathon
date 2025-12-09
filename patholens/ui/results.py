import streamlit as st
from PIL import Image

def display_metrics(results):
    """Render the top-level metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Cancer Probability",
            f"{results['heatmap_stats']['mean_probability']:.1%}"
        )

    with col2:
        st.metric(
            "Malignant Cells",
            f"{results['malignant_cell_count']}",
            delta=f"±{results['malignant_cell_uncertainty']:.0%}"
        )

    with col3:
        st.metric(
            "Patches Analysed",
            f"{results['num_patches']}"
        )

    with col4:
        st.metric(
            "Nuclei Detected",
            f"{results['nuclei_stats']['count']}"
        )
    st.markdown("---")

def render_overview_tab(results, original_image):
    """Render Tab 1: Images and Heatmaps with Toggle."""
    display_metrics(results)

    st.markdown("### Heatmap Analysis")
    
    # --- Toggle UI ---
    col_ctrl1, col_ctrl2 = st.columns([1, 3])
    with col_ctrl1:
        # Default to Absolute if key exists, otherwise Relative
        heatmap_mode = st.radio(
            "Visualization Mode",
            ["Enhanced (Relative)", "True Risk (Absolute)"],
            help="Enhanced: Stretches contrast so hotspots are visible even if low probability.\nTrue Risk: Shows raw probability (may be invisible if risk is low)."
        )
    
    # Determine which image to show
    # We check if the 'relative' key exists (added in pipeline update)
    if "Enhanced" in heatmap_mode and 'heatmap_overlay_relative' in results:
        display_img = results['heatmap_overlay_relative']
        caption = "Enhanced Heatmap (Contrast Boosted)"
    else:
        # Fallback to standard overlay
        display_img = results['heatmap_overlay']
        caption = "True Probability Heatmap (0-100% Scale)"

    # --- Image Display ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(original_image, use_container_width=True)

    with col2:
        st.markdown(f"**{caption}**")
        st.image(display_img, use_container_width=True)

    # Colour scale legend
    st.markdown("""
    <div style="display: flex; align-items: center; margin: 1rem 0;">
        <span style="margin-right: 1rem; font-size:0.8rem;">Low</span>
        <div style="flex: 1; height: 12px; background: linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #ffffbf, #fee090, #fdae61, #f46d43, #d73027, #a50026); border-radius: 4px;"></div>
        <span style="margin-left: 1rem; font-size:0.8rem;">High</span>
    </div>
    """, unsafe_allow_html=True)

def render_detailed_analysis_tab(results):
    """Render Tab 2: Grades and Statistics."""
    st.markdown("### Tumour Grade Estimation")
    
    grade = results['grade_estimate']
    col1, col2 = st.columns([2, 1])

    with col1:
        # Grade probability bars
        st.markdown(f"**Grade 1** (Well-differentiated): {grade.grade_1:.1%}")
        st.progress(grade.grade_1)

        st.markdown(f"**Grade 2** (Moderately-differentiated): {grade.grade_2:.1%}")
        st.progress(grade.grade_2)

        st.markdown(f"**Grade 3** (Poorly-differentiated): {grade.grade_3:.1%}")
        st.progress(grade.grade_3)

    with col2:
        st.markdown("#### Primary Estimate")
        st.markdown(f"## Grade {grade.primary_grade}")
        st.markdown(f"Confidence: {grade.confidence:.1%}")

    st.markdown("---")
    st.markdown("### Detailed Statistics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Heatmap Statistics")
        stats = results['heatmap_stats']
        st.markdown(f"- Mean probability: {stats['mean_probability']:.2%}")
        st.markdown(f"- Max probability: {stats['max_probability']:.2%}")
        st.markdown(f"- High-risk fraction: {stats['high_risk_fraction']:.2%}")
        st.markdown(f"- Moderate-risk fraction: {stats['moderate_risk_fraction']:.2%}")
        st.markdown(f"- Low-risk fraction: {stats['low_risk_fraction']:.2%}")

    with col2:
        st.markdown("#### Nuclear Features")
        nstats = results['nuclei_stats']
        gf = results['grading_features']
        st.markdown(f"- Total nuclei: {nstats['count']}")
        st.markdown(f"- Mean area: {nstats['mean_area']:.1f} px")
        st.markdown(f"- Pleomorphism score: {gf.nuclear_pleomorphism_score:.2f}")
        st.markdown(f"- Atypia score: {gf.nuclear_atypia_score:.2f}")
        st.markdown(f"- Texture heterogeneity: {gf.texture_heterogeneity:.2f}")

def render_interpretation_tab(results):
    """Render Tab 3: AI Interpretation Cards and Flow."""
    st.markdown("### Analysis Interpretation")

    # Extract key metrics
    mean_prob = results['heatmap_stats']['mean_probability']
    grade = results['grade_estimate']
    gf = results['grading_features']

    # Risk level determination
    if mean_prob < 0.3:
        risk_level, risk_colour, risk_bg = "Low", "#48bb78", "#f0fff4"
    elif mean_prob < 0.6:
        risk_level, risk_colour, risk_bg = "Moderate", "#ecc94b", "#fffff0"
    else:
        risk_level, risk_colour, risk_bg = "Elevated", "#f56565", "#fff5f5"

    # Key Findings Cards
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
        <div style="background: {risk_bg}; border: 1px solid {risk_colour}; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Risk Assessment</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: {risk_colour}; margin-bottom: 0.25rem;">{risk_level}</div>
            <div style="font-size: 0.9rem; color: #4a5568;">{mean_prob:.1%} probability</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Predicted Grade</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #2d3748; margin-bottom: 0.25rem;">Grade {grade.primary_grade}</div>
            <div style="font-size: 0.9rem; color: #4a5568;">{grade.confidence:.0%} confidence</div>
        </div>
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Cellular Analysis</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #2d3748; margin-bottom: 0.25rem;">{results['malignant_cell_count']}</div>
            <div style="font-size: 0.9rem; color: #4a5568;">suspected malignant cells</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Flow Diagram
    st.markdown("""
    <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem; margin: 2rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="font-size: 1rem; font-weight: 600; color: #2d3748; margin-bottom: 1.5rem; text-align: center;">Analysis Pipeline</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative;">
            <div style="position: absolute; top: 25px; left: 50px; right: 50px; height: 2px; background: #e2e8f0; z-index: 0;"></div>
            <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🔍</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748;">1. Scan</div>
            </div>
            <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🤖</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748;">2. Detect</div>
            </div>
            <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🔬</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748;">3. Segment</div>
            </div>
            <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">📊</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748;">4. Grade</div>
            </div>
            <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                <div style="width: 50px; height: 50px; background: #3182ce; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; color: white; box-shadow: 0 4px 6px rgba(49, 130, 206, 0.3);">📋</div>
                <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748;">5. Report</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Detailed Findings Columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Nuclear Morphology</div>
        """, unsafe_allow_html=True)
        
        # Helper to draw bars
        def draw_stat_bar(label, value):
            pct = value * 100
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>{label}</span>
                        <span>{value:.2f}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #4a5568; height: 100%; width: {pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        draw_stat_bar("Pleomorphism", gf.nuclear_pleomorphism_score)
        draw_stat_bar("Atypia", gf.nuclear_atypia_score)
        draw_stat_bar("Heterogeneity", gf.texture_heterogeneity)
        
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Grade Distribution</div>
        """, unsafe_allow_html=True)

        def draw_grade_bar(label, value, color):
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>{label}</span>
                        <span>{value:.1%}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: {color}; height: 100%; width: {value * 100}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        draw_grade_bar("Grade 1 (Well-diff.)", grade.grade_1, "#48bb78")
        draw_grade_bar("Grade 2 (Mod.-diff.)", grade.grade_2, "#ecc94b")
        draw_grade_bar("Grade 3 (Poorly-diff.)", grade.grade_3, "#f56565")

        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("View AI Interpretation", expanded=True):
        st.markdown(f"""
        <div style="background: #f7fafc; border-radius: 8px; padding: 1rem; font-size: 0.85rem; line-height: 1.6; color: #4a5568;">
        {results['explanation']}
        </div>
        """, unsafe_allow_html=True)

def render_results_dashboard(results, original_image):
    """Main entry point to display results."""
    st.markdown("---")
    st.markdown("## Analysis Results")

    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "AI Interpretation"])

    with tab1:
        render_overview_tab(results, original_image)
    
    with tab2:
        render_detailed_analysis_tab(results)

    with tab3:
        render_interpretation_tab(results)