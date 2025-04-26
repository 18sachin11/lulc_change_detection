import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import pandas as pd
import io
from pyproj import Transformer

# App title
st.title('🌎 Land Use Land Cover (LULC) Change Detection App')

st.markdown("""
This app detects changes between two Land Use Land Cover (LULC) raster files (.tif format).

**Features:**
- 📂 Upload raster files for two different years
- 🎯 Analyze all transition classes
- 🗺️ Visualize transition map with latitude-longitude grids, north arrow, scale bar, and bottom legends
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
            crs1 = src1.crs
            transform1 = src1.transform
            nodata1 = src1.nodata

        with rasterio.open(uploaded_file_2) as src2:
            land_cover_2 = src2.read(1)
            profile2 = src2.profile
            crs2 = src2.crs
            transform2 = src2.transform
            nodata2 = src2.nodata

        if land_cover_1.shape != land_cover_2.shape:
            st.error('❌ Error: Uploaded TIFF files do not match in shape or resolution.')
        else:
            if nodata1 is None:
                nodata1 = 0
            if nodata2 is None:
                nodata2 = 0

            valid_mask = (land_cover_1 != nodata1) & (land_cover_2 != nodata2)

            # Apply mask
            land_cover_1_valid = np.where(valid_mask, land_cover_1, np.nan)
            land_cover_2_valid = np.where(valid_mask, land_cover_2, np.nan)

            # Encode transitions
            transition_map = land_cover_1_valid * 100 + land_cover_2_valid
            transition_map = np.where(np.isnan(transition_map), np.nan, transition_map)

            # Find unique transitions
            transitions_unique = np.unique(transition_map[~np.isnan(transition_map)]).astype(int)
            st.success(f'✅ Unique transitions detected: {len(transitions_unique)}')

            # Create mapping for plotting
            transition_code_to_index = {code: idx for idx, code in enumerate(transitions_unique)}
            index_to_transition_code = {idx: code for idx, code in enumerate(transitions_unique)}

            transition_mapped = np.copy(transition_map)
            for code, idx in transition_code_to_index.items():
                transition_mapped[transition_map == code] = idx

            # --- Visualization ---
            st.subheader('🗺️ Change Transition Map')

            cmap = cm.get_cmap('nipy_spectral', len(transitions_unique))

            fig, ax = plt.subplots(figsize=(14, 10))

            img = ax.imshow(transition_mapped, cmap=cmap, interpolation='nearest')

            ax.set_title('Transition Map (From ➔ To)', fontsize=16)

            # Generate real-world coordinates
            nrows, ncols = transition_mapped.shape
            cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
            xs, ys = rasterio.transform.xy(transform1, rows, cols, offset='center')
            xs = np.array(xs)
            ys = np.array(ys)

            # Setup transformer
            transformer = Transformer.from_crs(crs1, "EPSG:4326", always_xy=True)
            x_deg, y_deg = transformer.transform(xs, ys)

            # Set x and y axis ticks
            ax.set_xticks(np.linspace(0, ncols-1, num=6))
            ax.set_xticklabels(["{:.4f}".format(val) for val in np.linspace(np.min(x_deg), np.max(x_deg), num=6)])

            ax.set_yticks(np.linspace(0, nrows-1, num=6))
            ax.set_yticklabels(["{:.4f}".format(val) for val in np.linspace(np.max(y_deg), np.min(y_deg), num=6)])

            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)

            # Add grids
            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5)
            ax.minorticks_on()

            # Add North Arrow
            ax.annotate('N', xy=(0.05, 0.95), xytext=(0.05, 0.95),
                        textcoords='axes fraction', fontsize=16,
                        ha='center', va='center',
                        arrowprops=dict(facecolor='black', width=5, headwidth=15))

            # Add Scale Bar
            scalebar_length_deg = 0.1
            pixel_width_deg = (np.max(x_deg) - np.min(x_deg)) / ncols
            scalebar_pixels = scalebar_length_deg / pixel_width_deg

            ax.plot([50, 50+scalebar_pixels], [nrows-20, nrows-20], color='black', lw=4)
            ax.text(50, nrows-10, f"{scalebar_length_deg}°", fontsize=10, ha='left')

            # Create bottom legend
            labels = [f"{str(code)[:-2]} ➔ {str(code)[-2:]}" for code in transitions_unique]
            colors = [cmap(i / len(transitions_unique)) for i in range(len(transitions_unique))]
            patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i],
                                label="{:}".format(labels[i]))[0] for i in range(len(labels))]

            leg = ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.55),
                            fancybox=True, shadow=True, ncol=5, title="Transitions")
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
