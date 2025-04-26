import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import io

# App title
st.title('🌎 Land Use Land Cover (LULC) Change Detection App')

st.markdown("""
This app detects changes between two Land Use Land Cover (LULC) raster files (.tif format).
- 📂 Upload raster files for two different years
- 🎯 Analyze all transition classes
- 🗺️ Visualize the transition map with legend, grid, and north arrow
- 📋 See and download the change summary table
- 📥 Download transition raster
""")

# Upload TIFFs
uploaded_file_1 = st.file_uploader("Upload TIFF file for Year 1", type=["tif", "tiff"])
uploaded_file_2 = st.file_uploader("Upload TIFF file for Year 2", type=["tif", "tiff"])

if uploaded_file_1 and uploaded_file_2:
    try:
        with rasterio.open(uploaded_file_1) as src1:
            land_cover_1 = src1.read(1)
            profile1 = src1.profile
        
        with rasterio.open(uploaded_file_2) as src2:
            land_cover_2 = src2.read(1)
            profile2 = src2.profile

        if land_cover_1.shape != land_cover_2.shape:
            st.error('❌ Error: Uploaded TIFF files do not match in shape or resolution.')
        else:
            # Find unique classes
            unique_classes_year1 = np.unique(land_cover_1)
            unique_classes_year2 = np.unique(land_cover_2)
            unique_classes = np.unique(np.concatenate((unique_classes_year1, unique_classes_year2)))

            st.write('Unique Classes Detected:', unique_classes)

            # Encode transitions
            transition_map = land_cover_1 * 100 + land_cover_2

            # Get all unique transitions
            transitions = np.unique(transition_map)

            # --- Visualization ---
            st.subheader('🗺️ Change Transition Map')

            fig, ax = plt.subplots(figsize=(12, 8))
            cmap = plt.cm.get_cmap('tab20', len(transitions))
            norm = mcolors.BoundaryNorm(boundaries=np.arange(transitions.min()-0.5, transitions.max()+1.5, 1), ncolors=len(transitions))

            img = ax.imshow(transition_map, cmap=cmap, interpolation='nearest')
            ax.set_title('Transition Map (From ➔ To)', fontsize=16)
            ax.set_xlabel('Column Index', fontsize=12)
            ax.set_ylabel('Row Index', fontsize=12)

            # Add grids
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5)
            ax.minorticks_on()

            # Add North Arrow
            ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.95),
                        textcoords='axes fraction', fontsize=16,
                        ha='center', va='center',
                        arrowprops=dict(facecolor='black', width=5, headwidth=15))

            # Add custom legend
            labels = [f"{str(val)[:1]} ➔ {str(val)[-1:]}" for val in transitions]
            colors = [cmap(i/len(transitions)) for i in range(len(transitions))]
            patches = [plt.plot([],[], marker="s", ms=10, ls="", mec=None, color=colors[i], 
                                label="{:}".format(labels[i]) )[0]  for i in range(len(labels))]
            leg = ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title="Transitions")
            plt.tight_layout()

            st.pyplot(fig)

            # --- Change Summary Table ---
            st.subheader('📋 Change Summary Table')

            # Flatten arrays
            land_cover_1_flat = land_cover_1.flatten()
            land_cover_2_flat = land_cover_2.flatten()

            # Create DataFrame
            change_data = pd.DataFrame({
                'From': land_cover_1_flat,
                'To': land_cover_2_flat
            })

            # Group and count
            change_summary = change_data.groupby(['From', 'To']).size().reset_index(name='Pixel Count')

            # Show table
            st.dataframe(change_summary)

            # --- Download options ---
            st.subheader('📥 Download Results')

            # Download transition raster
            buffer = io.BytesIO()
            with rasterio.open(
                buffer, 'w', driver='GTiff',
                height=transition_map.shape[0],
                width=transition_map.shape[1],
                count=1, dtype=transition_map.dtype,
                crs=profile1['crs'], transform=profile1['transform']
            ) as dst:
                dst.write(transition_map, 1)
            buffer.seek(0)

            st.download_button(
                label="Download Transition Raster (.tif)",
                data=buffer,
                file_name='transition_map.tif',
                mime='image/tiff'
            )

            # Download change table
            csv = change_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Change Summary Table (.csv)",
                data=csv,
                file_name='change_summary.csv',
                mime='text/csv'
            )

            st.success('✅ Analysis Complete!')

    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")

else:
    st.info('Please upload two TIFF files to begin.')

# Footer
st.markdown("---")
st.caption("Developed by [Your Name] | Powered by Streamlit 🚀")
