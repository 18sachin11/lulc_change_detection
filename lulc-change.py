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
- 🗓️ Specify custom years for each map
- 🎯 Analyze and visualize land cover for each year
- 🗺️ Visualize transition map with Dynamic World classes and unique transition legends
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
    0: '#419bdf',
    1: '#397d49',
    2: '#88b053',
    3: '#7a87c6',
    4: '#e49635',
    5: '#dfc35a',
    6: '#c4281b',
    7: '#a59b8f'
}

# User inputs for years
st.subheader('🗓️ Specify Years for Uploaded Maps')

year1 = st.text_input("Enter the year for the first uploaded TIFF (e.g., 2015):", value="Year 1")
year2 = st.text_input("Enter the year for the second uploaded TIFF (e.g., 2020):", value="Year 2")

# Upload TIFFs
uploaded_file_1 = st.file_uploader(f"Upload TIFF file for {year1}", type=["tif", "tiff"])
uploaded_file_2 = st.file_uploader(f"Upload TIFF file for {year2}", type=["tif", "tiff"])

# Define plot function
def plot_map(image_array, title, transform, crs, is_transition=False, transition_info=None):
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

    if is_transition and transition_info:
        transition_mapped = transition_info['mapped_array']
        colors = transition_info['colors']
        labels = transition_info['labels']
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        ax.imshow(transition_mapped, cmap=cmap, interpolation='nearest')
    else:
        cmap = plt.matplotlib.colors.ListedColormap([class_color_mapping.get(i, '#d3d3d3') for i in range(8)])
        ax.imshow(image_array, cmap=cmap, interpolation='nearest')
        labels = [class_label_mapping[i] for i in range(8)]
        colors = [class_color_mapping[i] for i in range(8)]

    ax.set_title(title, fontsize=16)

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

    # Legend
    ax_leg = fig.add_subplot(gs[1, 0])
    ax_leg.axis('off')
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)]
    ax_leg.legend(handles=patches, loc='center', fancybox=True, shadow=True, ncol=4, title="Classes" if not is_transition else "Transitions")

    plt.tight_layout()
    st.pyplot(fig)

# After file upload
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
            # Valid data
            if nodata1 is None:
                nodata1 = 0
            if nodata2 is None:
                nodata2 = 0

            valid_mask = (land_cover_1 != nodata1) & (land_cover_2 != nodata2)

            land_cover_1_valid = np.where(valid_mask, land_cover_1, np.nan)
            land_cover_2_valid = np.where(valid_mask, land_cover_2, np.nan)

            # Transition Map
            transition_map = land_cover_1_valid * 100 + land_cover_2_valid
            transition_map = np.where(np.isnan(transition_map), np.nan, transition_map)

            # --- Plot Year 1 Map
            st.subheader(f'🗺️ LULC Map for {year1}')
            plot_map(land_cover_1_valid, f'LULC Map for {year1}', transform1, crs1)

            # --- Plot Year 2 Map
            st.subheader(f'🗺️ LULC Map for {year2}')
            plot_map(land_cover_2_valid, f'LULC Map for {year2}', transform2, crs2)

            # --- Plot Transition Map
            transitions_unique = np.unique(transition_map[~np.isnan(transition_map)]).astype(int)
            transition_code_to_index = {code: idx for idx, code in enumerate(transitions_unique)}
            transition_mapped = np.copy(transition_map)
            for code, idx in transition_code_to_index.items():
                transition_mapped[transition_map == code] = idx

            transition_colors = []
            transition_labels = []
            for code in transitions_unique:
                code_str = str(code)
                if len(code_str) <= 2:
                    from_class = 0
                    to_class = int(code_str)
                else:
                    from_class = int(code_str[:-2])
                    to_class = int(code_str[-2:])
                color = class_color_mapping.get(to_class, '#d3d3d3')
                transition_colors.append(color)
                from_label = class_label_mapping.get(from_class, 'Unknown')
                to_label = class_label_mapping.get(to_class, 'Unknown')
                transition_labels.append(f"{from_label} ➔ {to_label}")

            st.subheader(f'🗺️ Transition Map ({year1} ➔ {year2})')
            transition_info = {
                'mapped_array': transition_mapped,
                'colors': transition_colors,
                'labels': transition_labels
            }
            plot_map(transition_map, f'Transition Map ({year1} ➔ {year2})', transform1, crs1,
                     is_transition=True, transition_info=transition_info)

            # --- Change Summary Table
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

            # --- Download Transition Raster and Table
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
    st.info('Please upload two TIFF files to continue.')

# Footer
st.markdown("---")
st.caption("Developed by Sachchidanand Singh | Powered by Streamlit 🚀")
