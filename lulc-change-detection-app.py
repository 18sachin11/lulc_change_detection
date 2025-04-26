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
- 🎯 Select specific classes to analyze
- 🗺️ Visualize the changes
- 📋 See and download the change summary table
- 📥 Download change detection raster
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
            # List unique classes
            unique_classes = np.unique(np.concatenate((land_cover_1.flatten(), land_cover_2.flatten())))
            selected_classes = st.multiselect(
                "Select classes to analyze (optional):", 
                options=list(unique_classes),
                default=list(unique_classes)
            )

            # Change Detection
            change_map = np.where(land_cover_1 != land_cover_2, land_cover_2, 0)

            # Filter selected classes
            if selected_classes:
                mask_selected = np.isin(change_map, selected_classes)
                change_map_filtered = np.where(mask_selected, change_map, 0)
            else:
                change_map_filtered = change_map

            # --- Visualization ---
            st.subheader('🗺️ Change Detection Map')

            cmap = plt.cm.get_cmap('tab20', len(unique_classes))
            norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, len(unique_classes)+0.5, 1), ncolors=len(unique_classes))

            fig, ax = plt.subplots(figsize=(10, 6))
            img = ax.imshow(change_map_filtered, cmap=cmap, norm=norm)
            cbar = plt.colorbar(img, ax=ax, ticks=np.arange(len(unique_classes)))
            cbar.ax.set_yticklabels([str(cls) for cls in unique_classes])
            ax.set_title('Detected Changes')
            ax.axis('off')
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

            # Filter only changes
            change_data = change_data[change_data['From'] != change_data['To']]

            # Group and count
            change_summary = change_data.groupby(['From', 'To']).size().reset_index(name='Pixel Count')

            # Show table
            st.dataframe(change_summary)

            # --- Download options ---
            st.subheader('📥 Download Results')

            # Download change raster
            buffer = io.BytesIO()
            with rasterio.open(
                buffer, 'w', driver='GTiff',
                height=change_map_filtered.shape[0],
                width=change_map_filtered.shape[1],
                count=1, dtype=change_map_filtered.dtype,
                crs=profile1['crs'], transform=profile1['transform']
            ) as dst:
                dst.write(change_map_filtered, 1)
            buffer.seek(0)

            st.download_button(
                label="Download Change Detection Raster (.tif)",
                data=buffer,
                file_name='change_detection.tif',
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
