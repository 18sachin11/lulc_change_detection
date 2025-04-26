import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import io

# App title
st.title('🌎 Land Use Land Cover (LULC) Change Detection App')

st.markdown("""
This app detects changes between two Land Use Land Cover (LULC) raster files (.tif format).

**Features:**
- 📂 Upload raster files for two different years
- 🎯 Analyze all transition classes
- 🗺️ Visualize transition map with legends, grids, and north arrow
- 📋 View and download change summary table
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
            nodata1 = src1.nodata

        with rasterio.open(uploaded_file_2) as src2:
            land_cover_2 = src2.read(1)
            profile2 = src2.profile
            nodata2 = src2.nodata

        if land_cover_1.shape != land_cover_2.shape:
            st.error('❌ Error: Uploaded TIFF files do not match in shape or resolution.')
        else:
            # Handle nodata
            if nodata1 is None:
                nodata1 = 0
            if nodata2 is None:
                nodata2 = 0

            valid_mask = (land_cover_1 != nodata1) & (land_cover_2 != nodata2)

            # Mask invalid pixels
            land_cover_1_valid = np.where(valid_mask, land_cover_1, np.nan)
            land_cover_2_valid = np.where(valid_mask, land_cover_2, np.nan)

            # Encode transitions
            transition_map = land_cover_1_valid * 100 + land_cover_2_valid
            transition_map = np.where(np.isnan(transition_map), np.nan, transition_map)

            # Unique transitions
            transitions = np.unique(transition_map[~np.isnan(transition_map)]).astype(int)

            st.success(f'✅ Unique transitions detected: {len(transitions)}')

            # --- Visualization ---
            st.subheader('🗺️ Change Transition Map')

            cmap = cm.get_cmap('nipy_spectral', len(transitions))
            boundaries = np.arange(min(transitions) - 0.5, max(transitions) + 1.5, 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N)

            fig, ax = plt.subplots(figsize=(12, 8))
            img = ax.imshow(transition_map, cmap=cmap, norm=norm, interpolation='nearest')

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

            # Custom Legend
            labels = [f"{str(val)[:-2]} ➔ {str(val)[-2:]}" for val in transitions]
            colors = [cmap(i / len(transitions)) for i in range(len(transitions))]
            patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i],
                                label="{:}".format(labels[i]))[0] for i in range(len(labels))]
            ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left',
                      borderaxespad=0., title="Transitions")
            plt.tight_layout()

            st.pyplot(fig)

            # --- Change Summary Table ---
            st.subheader('📋 Change Summary Table')

            # Flatten valid arrays
            land_cover_1_flat = land_cover_1_valid.flatten()
            land_cover_2_flat = land_cover_2_valid.flatten()

            valid_indices = ~np.isnan(land_cover_1_flat) & ~np.isnan(land_cover_2_flat)
            land_cover_1_flat = land_cover_1_flat[valid_indices]
            land_cover_2_flat = land_cover_2_flat[valid_indices]

            change_data = pd.DataFrame({
                'From': land_cover_1_flat.astype(int),
                'To': land_cover_2_flat.astype(int)
            })

            change_summary = change_data.groupby(['From', 'To']).size().reset_index(name='Pixel Count')

            st.dataframe(change_summary)

            # --- Download options ---
            st.subheader('📥 Download Results')

            # Prepare raster for download
            nodata_value = -9999
            transition_map_final = np.where(np.isnan(transition_map), nodata_value, transition_map)

            buffer = io.BytesIO()
            with rasterio.open(
                buffer, 'w', driver='GTiff',
                height=transition_map_final.shape[0],
                width=transition_map_final.shape[1],
                count=1, dtype='float32',
                crs=profile1['crs'], transform=profile1['transform'],
                nodata=nodata_value
            ) as dst:
                dst.write(transition_map_final.astype('float32'), 1)
            buffer.seek(0)

            st.download_button(
                label="Download Transition Raster (.tif)",
                data=buffer,
                file_name='transition_map.tif',
                mime='image/tiff'
            )

            # Download table as CSV
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
