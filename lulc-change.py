import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
- 🗺️ Visualize transition map with Dynamic World colors, 2-decimal lat/lon grids, north arrow, bottom legends, and proper scale bar
- 📋 View and download change summary table
- 📥 Download transition raster
""")

# Dynamic World Class Colors
class_color_mapping = {
    0: '#419bdf', # Water
    1: '#397d49', # Trees
    2: '#88b053', # Grass
    3: '#7a87c6', # Flooded Vegetation
    4: '#e49635', # Crops
    5: '#dfc35a', # Shrub and Scrub
    6: '#c4281b', # Built Area
    7: '#a59b8f'  # Bare Ground
}

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

            land_cover_1_valid = np.where(valid_mask, land_cover_1, np.nan)
            land_cover_2_valid = np.where(valid_mask, land_cover_2, np.nan)

            transition_map = land_cover_1_valid * 100 + land_cover_2_valid
            transition_map = np.where(np.isnan(transition_map), np.nan, transition_map)

            transitions_unique = np.unique(transition_map[~np.isnan(transition_map)]).astype(int)
            st.success(f'✅ Unique transitions detected: {len(transitions_unique)}')

            transition_code_to_index = {code: idx for idx, code in enumerate(transitions_unique)}
            index_to_transition_code = {idx: code for idx, code in enumerate(transitions_unique)}

            transition_mapped = np.copy(transition_map)
            for code, idx in transition_code_to_index.items():
                transition_mapped[transition_map == code] = idx

            # --- Visualization ---
            st.subheader('🗺️ Change Transition Map')

            fig, ax = plt.subplots(figsize=(14, 10))

            # Create a custom colormap based on transitions
            colors = []
            for code in transitions_unique:
                from_class = int(str(code)[:-2])
                to_class = int(str(code)[-2:])
                color = class_color_mapping.get(to_class, '#d3d3d3') # fallback light grey
                colors.append(color)

            cmap = plt.matplotlib.colors.ListedColormap(colors)

            img = ax.imshow(transition_mapped, cmap=cmap, interpolation='nearest')

            ax.set_title('Transition Map (From ➔ To)', fontsize=16)

            # Generate real-world coordinates
            nrows, ncols = transition_mapped.shape
            cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
            xs, ys = rasterio.transform.xy(transform1, rows, cols, offset='center')
            xs = np.array(xs)
            ys = np.array(ys)

            transformer = Transformer.from_crs(crs1, "EPSG:4326", always_xy=True)
            x_deg, y_deg = transformer.transform(xs, ys)

            # Set ticks (formatted to 2 decimal places)
            ax.set_xticks(np.linspace(0, ncols-1, num=6))
            ax.set_xticklabels(["{:.2f}".format(val) for val in np.linspace(np.min(x_deg), np.max(x_deg), num=6)])

            ax.set_yticks(np.linspace(0, nrows-1, num=6))
            ax.set_yticklabels(["{:.2f}".format(val) for val in np.linspace(np.max(y_deg), np.min(y_deg), num=6)])

            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)

            ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5)
            ax.minorticks_on()

            # --- Fancy North Arrow ---
            ax.annotate('N', xy=(0.97, 0.98), xycoords='axes fraction',
                        fontsize=16, fontweight='bold', ha='center')
            ax.annotate('↑', xy=(0.97, 0.94), xycoords='axes fraction',
                        fontsize=20, ha='center')

            # --- Bottom Legend ---
            patches = []
            for code, color in zip(transitions_unique, colors):
                label = f"{str(code)[:-2]} ➔ {str(code)[-2:]}"
                patches.append(mpatches.Patch(color=color, label=label))

            leg = ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.55),
                            fancybox=True, shadow=True, ncol=5, title="Transitions")

            # --- Scale Bar below legend ---
            fig.subplots_adjust(bottom=0.4)  # Give extra space
            scalebar_ax = fig.add_axes([0.4, 0.05, 0.2, 0.02])
            scalebar_ax.axis('off')

            scalebar_ax.plot([0, 1], [0.5, 0.5], color='black', lw=6)
            scalebar_ax.text(0, 0.8, '0°', fontsize=10, va='bottom', ha='center')
            scalebar_ax.text(1, 0.8, '0.1°', fontsize=10, va='bottom', ha='center')
            scalebar_ax.set_xlim(0, 1)
            scalebar_ax.set_ylim(0, 1)

            plt.tight_layout()

            st.pyplot(fig)

            # --- Change Summary Table ---
            st.subheader('📋 Change Summary Table')

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
