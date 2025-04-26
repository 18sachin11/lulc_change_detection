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
- 🎯 Analyze and visualize land cover for Year 1, Year 2
- 🗺️ Visualize transition map with Dynamic World class names (FROM ➔ TO)
- 📋 View and download change summary table
- 📥 Download transition raster
""")

# Dynamic World Class Labels and Colors
class_label_mapping = {
    0: 'Water',
    1: 'Trees',
    2: 'Grass',
    3: 'Flooded Vegetation',
    4: 'Crops',
    5: 'Shrub and Scrub',
    6: 'Built Area',
    7: 'Bare Ground'
}

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

# Define function to plot LULC or Transition map
def plot_map(image_array, title, transform, crs, class_label_mapping, class_color_mapping, is_transition=False):
    nrows, ncols = image_array.shape
    cols, rows = np.meshgrid(np.arange(ncols), np.arange(nrows))
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    xs = np.array(xs)
    ys = np.array(ys)

    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x_deg, y_deg = transformer.transform(xs, ys)

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1])

    ax = fig.add_subplot(gs[0, 0])

    if is_transition:
        # Transition Map
        transitions_unique = np.unique(image_array[~np.isnan(image_array)]).astype(int)
        transition_code_to_index = {code: idx for idx, code in enumerate(transitions_unique)}
        transition_mapped = np.copy(image_array)
        for code, idx in transition_code_to_index.items():
            transition_mapped[image_array == code] = idx

        # Colors and labels
        colors = []
        labels = []
        for code in transitions_unique:
            code_str = str(code)
            if len(code_str) <= 2:
                from_class = 0
                to_class = int(code_str)
            else:
                from_class = int(code_str[:-2])
                to_class = int(code_str[-2:])

            color = class_color_mapping.get(to_class, '#d3d3d3')
            colors.append(color)

            from_label = class_label_mapping.get(from_class, 'Unknown')
            to_label = class_label_mapping.get(to_class, 'Unknown')
            labels.append(f"{from_label} ➔ {to_label}")

        cmap = plt.matplotlib.colors.ListedColormap(colors)
        ax.imshow(transition_mapped, cmap=cmap, interpolation='nearest')

    else:
        # LULC Map
        cmap = plt.matplotlib.colors.ListedColormap([class_color_mapping.get(i, '#d3d3d3') for i in range(8)])
        ax.imshow(image_array, cmap=cmap, interpolation='nearest')
        labels = [class_label_mapping[i] for i in range(8)]
        colors = [class_color_mapping[i] for i in range(8)]

    ax.set_title(title, fontsize=16)

    # Axis settings
    ax.set_xticks(np.linspace(0, ncols-1, num=6))
    ax.set_xticklabels(["{:.2f}".format(val) for val in np.linspace(np.min(x_deg), np.max(x_deg), num=6)])
    ax.set_yticks(np.linspace(0, nrows-1, num=6))
    ax.set_yticklabels(["{:.2f}".format(val) for val in np.linspace(np.max(y_deg), np.min(y_deg), num=6)])
    ax.set_xlabel('Longitude (°)', fontsize=12)
    ax.set_ylabel('Latitude (°)', fontsize=12)
    ax.grid(which='both', color='grey', linestyle='--', linewidth=0.5)
    ax.minorticks_on()

    # North Arrow
    ax.annotate('N', xy=(0.97, 0.98), xycoords='axes fraction',
                fontsize=16, fontweight='bold', ha='center')
    ax.annotate('↑', xy=(0.97, 0.94), xycoords='axes fraction',
                fontsize=20, ha='center')

    # Legend axis
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis('off')
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax_leg.legend(handles=patches, loc='center', fancybox=True, shadow=True, ncol=4, title="Classes")

    plt.tight_layout()
    st.pyplot(fig)

# Process after both uploads
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
            # Apply valid mask if needed
            if nodata1 is None:
                nodata1 = 0
            if nodata2 is None:
                nodata2 = 0

            valid_mask = (land_cover_1 != nodata1) & (land_cover_2 != nodata2)

            land_cover_1_valid = np.where(valid_mask, land_cover_1, np.nan)
            land_cover_2_valid = np.where(valid_mask, land_cover_2, np.nan)

            transition_map = land_cover_1_valid * 100 + land_cover_2_valid
            transition_map = np.where(np.isnan(transition_map), np.nan, transition_map)

            st.subheader('🗺️ Year 1 LULC Map')
            plot_map(land_cover_1_valid, 'Year 1 LULC Map', transform1, crs1,
                     class_label_mapping, class_color_mapping, is_transition=False)

            st.subheader('🗺️ Year 2 LULC Map')
            plot_map(land_cover_2_valid, 'Year 2 LULC Map', transform2, crs2,
                     class_label_mapping, class_color_mapping, is_transition=False)

            st.subheader('🗺️ Transition Map (From ➔ To)')
            plot_map(transition_map, 'Transition Map (From ➔ To)', transform1, crs1,
                     class_label_mapping, class_color_mapping, is_transition=True)

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
