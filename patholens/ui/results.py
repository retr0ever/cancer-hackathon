import streamlit as st
from html import escape

def display_metrics(results):
    """Render the top-level metrics row."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Cancer Probability",
            f"{results['heatmap_stats']['mean_probability']:.1%}",
            delta=None
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

def render_results_dashboard(results, original_image):
    """Main entry point to display results."""
    st.markdown("---")
    st.markdown("## Analysis Results")

    # Create tabs for better organization
    tab1, tab2, tab3 = st.tabs(["Overview", "Detailed Analysis", "AI Interpretation"])

    with tab1:
        display_metrics(results)
        st.markdown("---")
        
        # Image comparison
        st.markdown("### Heatmap Overlay")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Original Image**")
            st.image(original_image, use_container_width=True)

        with col2:
            st.markdown("**Cancer Probability Heatmap**")
            st.image(results['heatmap_overlay'], use_container_width=True)

        # Colour scale legend
        st.markdown("""
        <div style="display: flex; align-items: center; margin: 1rem 0;">
            <span style="margin-right: 1rem;">Low Risk</span>
            <div style="flex: 1; height: 20px; background: linear-gradient(to right, #313695, #4575b4, #74add1, #abd9e9, #e0f3f8, #ffffbf, #fee090, #fdae61, #f46d43, #d73027, #a50026); border-radius: 4px;"></div>
            <span style="margin-left: 1rem;">High Risk</span>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        # Grade estimation
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
            st.markdown("#### Most Probable Grade")
            st.markdown(f"## Grade {grade.primary_grade}")
            st.markdown(f"Probability gap vs. next grade: {grade.confidence:.1%}")
            st.caption("Indicates how much more this grade is supported than the runner-up; it is not a model-wide accuracy metric.")

        st.markdown("---")

        # Detailed statistics
        st.markdown("### Detailed Analysis")

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

    with tab3:
        # Enhanced Analysis Interpretation UI
        st.markdown("### Analysis Interpretation")

        # Extract key metrics for display
        mean_prob = results['heatmap_stats']['mean_probability']
        grade = results['grade_estimate']
        gf = results['grading_features']
        nstats = results['nuclei_stats']

        # Risk level determination
        if mean_prob < 0.3:
            risk_level = "Low"
            risk_colour = "#48bb78"
            risk_bg = "#f0fff4"
        elif mean_prob < 0.6:
            risk_level = "Moderate"
            risk_colour = "#ecc94b"
            risk_bg = "#fffff0"
        else:
            risk_level = "Elevated"
            risk_colour = "#f56565"
            risk_bg = "#fff5f5"

        # Key Findings Cards
        st.markdown(f"""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin: 1.5rem 0;">
            <div style="background: {risk_bg}; border: 1px solid {risk_colour}; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Risk Assessment</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: {risk_colour}; margin-bottom: 0.25rem;">{risk_level}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">{mean_prob:.1%} probability</div>
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Most Probable Grade</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #2d3748; margin-bottom: 0.25rem;">Grade {grade.primary_grade}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">{grade.confidence:.0%} probability gap vs. next grade</div>
            </div>
            <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; font-weight: 600;">Cellular Analysis</div>
                <div style="font-size: 1.75rem; font-weight: 700; color: #2d3748; margin-bottom: 0.25rem;">{results['malignant_cell_count']}</div>
                <div style="font-size: 0.9rem; color: #4a5568;">suspected malignant cells</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Analysis Flow Diagram (Human Readable)
        st.markdown("""
        <div style="background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 2rem; margin: 2rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="font-size: 1rem; font-weight: 600; color: #2d3748; margin-bottom: 1.5rem; text-align: center;">Analysis Pipeline</div>
            <div style="display: flex; justify-content: space-between; align-items: flex-start; position: relative;">
                <!-- Connecting Line -->
                <div style="position: absolute; top: 25px; left: 50px; right: 50px; height: 2px; background: #e2e8f0; z-index: 0;"></div>
                
                <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                    <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🔍</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748; margin-bottom: 0.25rem;">1. Scan</div>
                    <div style="font-size: 0.75rem; color: #718096; line-height: 1.4;">Scanning image for tissue regions</div>
                </div>
                <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                    <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🤖</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748; margin-bottom: 0.25rem;">2. Detect</div>
                    <div style="font-size: 0.75rem; color: #718096; line-height: 1.4;">AI identifies potential cancer cells</div>
                </div>
                <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                    <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">🔬</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748; margin-bottom: 0.25rem;">3. Segment</div>
                    <div style="font-size: 0.75rem; color: #718096; line-height: 1.4;">Isolating individual cell nuclei</div>
                </div>
                <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                    <div style="width: 50px; height: 50px; background: #ebf8ff; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; border: 2px solid #bee3f8;">📊</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748; margin-bottom: 0.25rem;">4. Grade</div>
                    <div style="font-size: 0.75rem; color: #718096; line-height: 1.4;">Assessing tumour severity</div>
                </div>
                <div style="text-align: center; z-index: 1; background: white; padding: 0 10px; width: 18%;">
                    <div style="width: 50px; height: 50px; background: #3182ce; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 1rem auto; font-size: 1.5rem; color: white; box-shadow: 0 4px 6px rgba(49, 130, 206, 0.3);">📋</div>
                    <div style="font-weight: 600; font-size: 0.9rem; color: #2d3748; margin-bottom: 0.25rem;">5. Report</div>
                    <div style="font-size: 0.75rem; color: #718096; line-height: 1.4;">Compiling final risk assessment</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Detailed Findings in columns
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Nuclear Morphology</div>
            """, unsafe_allow_html=True)

            # Pleomorphism indicator
            pleo_pct = gf.nuclear_pleomorphism_score * 100
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Pleomorphism</span>
                        <span>{gf.nuclear_pleomorphism_score:.2f}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #4a5568; height: 100%; width: {pleo_pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Atypia indicator
            atypia_pct = gf.nuclear_atypia_score * 100
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Atypia</span>
                        <span>{gf.nuclear_atypia_score:.2f}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #4a5568; height: 100%; width: {atypia_pct}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Heterogeneity indicator
            het_pct = gf.texture_heterogeneity * 100
            st.markdown(f"""
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Heterogeneity</span>
                        <span>{gf.texture_heterogeneity:.2f}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #4a5568; height: 100%; width: {het_pct}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div style="background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.25rem; height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 600; color: #4a5568; margin-bottom: 1rem;">Grade Distribution</div>
            """, unsafe_allow_html=True)

            # Grade 1
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Grade 1 (Well-diff.)</span>
                        <span>{grade.grade_1:.1%}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #48bb78; height: 100%; width: {grade.grade_1 * 100}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Grade 2
            st.markdown(f"""
                <div style="margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Grade 2 (Mod.-diff.)</span>
                        <span>{grade.grade_2:.1%}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #ecc94b; height: 100%; width: {grade.grade_2 * 100}%;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Grade 3
            st.markdown(f"""
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #718096; margin-bottom: 0.25rem;">
                        <span>Grade 3 (Poorly-diff.)</span>
                        <span>{grade.grade_3:.1%}</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: #f56565; height: 100%; width: {grade.grade_3 * 100}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # AI Interpretation (collapsible)
        with st.expander("View AI Interpretation", expanded=True):
            explanation_text = results.get('explanation') or "No AI interpretation available for this case."
            formatted_explanation = escape(explanation_text).replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="background: #f7fafc; border-radius: 8px; padding: 1rem; font-size: 0.85rem; line-height: 1.6; color: #4a5568;">
                    {formatted_explanation}
                </div>
                """,
                unsafe_allow_html=True
            )
