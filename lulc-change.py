import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import io
from pyproj import Transformer
import matplotlib.cm as cm

# Title
st.title('🌎 Land Use Land Cover (LULC) Change Detection and Prediction App')

st.markdown("""
This app allows you to:
- 📂 Upload two LULC raster files (TIFF)
- 🗓️ Specify years for each file
- 🗺️ View LULC maps and transitions
- 📈 Predict future LULC using Markov Chain
- 📥 Download results
""")

# Dynamic World Class Labels and Colors
class_label_mapping = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation',
    4: 'Crops', 5: 'Shrub and Scrub', 6: 'Built Area', 7: 'Bare Ground'
}

class_color_mapping = {
    0: '#419bdf', 1: '#397d49', 2: '#88b053', 3: '#7a87c6',
    4: '#e49635', 5: '#dfc35a', 6: '#c4281b', 7: '#a59b8f'
}

# User input for years
st.subheader('🗓️ Specify Years')

year1 = st.text_input("Enter the year for the first uploaded TIFF (e.g., 2010):", value="Year 1")
year2 = st.text_input("Enter the year for the second uploaded TIFF (e.g., 2020):", value="Year 2")

# Upload files
uploaded_file_1 = st.file_uploader(f"Upload TIFF for {year1}", type=["tif", "tiff"])
uploaded_file_2 = st.file_uploader(f"Upload TIFF for {year2}", type=["tif", "tiff"])

# Function to plot map
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
        cmap = plt.matplotlib.colors.ListedColormap(transition_info['colors'])
        ax.imshow(transition_info['mapped_array'], cmap=cmap, interpolation='nearest')
        labels = transition_info['labels']
        colors = transition_info['colors']
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
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
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
    ax_leg.legend(handles=patches, loc='center', fancybox=True, shadow=True, ncol=3,
                  title="Transitions" if is_transition else "Classes")
    plt.tight_layout()
    st.pyplot(fig)

# Main Processing
if uploaded_file_1 and uploaded_file_2:
    try:
        with rasterio.open(uploaded_file_1) as src1:
            lc1 = src1.read(1)
            profile1 = src1.profile
            crs1 = src1.crs
            transform1 = src1.transform

        with rasterio.open(uploaded_file_2) as src2:
            lc2 = src2.read(1)
            profile2 = src2.profile
            crs2 = src2.crs
            transform2 = src2.transform

        if lc1.shape != lc2.shape:
            st.error('❌ Error: Files have different shapes!')
        else:
            valid_mask = (~np.isnan(lc1)) & (~np.isnan(lc2))

            # Plot maps
            st.subheader(f'🗺️ LULC Map for {year1}')
            plot_map(lc1, f'LULC Map for {year1}', transform1, crs1)

            st.subheader(f'🗺️ LULC Map for {year2}')
            plot_map(lc2, f'LULC Map for {year2}', transform2, crs2)

            # Transition Map
            transition_map = lc1 * 100 + lc2
            transitions_unique = np.unique(transition_map[valid_mask])

            transition_mapped = np.copy(transition_map)
            trans_idx = {code: idx for idx, code in enumerate(transitions_unique)}
            for code, idx in trans_idx.items():
                transition_mapped[transition_map == code] = idx

            cmap_transitions = cm.get_cmap('tab20', len(transitions_unique))
            transition_colors = [cmap_transitions(i) for i in range(len(transitions_unique))]
            transition_labels = []
            for code in transitions_unique:
                from_label = class_label_mapping.get(int(str(code)[:-2]), 'Unknown')
                to_label = class_label_mapping.get(int(str(code)[-2:]), 'Unknown')
                transition_labels.append(f"{from_label} ➔ {to_label}")

            st.subheader(f'🗺️ Transition Map ({year1} ➔ {year2})')
            transition_info = {'mapped_array': transition_mapped, 'colors': transition_colors, 'labels': transition_labels}
            plot_map(transition_map, f'Transition Map ({year1} ➔ {year2})', transform1, crs1, is_transition=True, transition_info=transition_info)

            # Change Summary Table
            st.subheader('📋 Change Summary Table')
            df_changes = pd.DataFrame({
                'From': (lc1[valid_mask]).astype(int),
                'To': (lc2[valid_mask]).astype(int)
            })
            summary = df_changes.groupby(['From', 'To']).size().reset_index(name='Count')
            st.dataframe(summary)

            # Future Prediction Section
            st.subheader('📈 Future Land Use Prediction (Markov Chain)')

            future_year = st.number_input(f'Enter year you want to predict (after {year2}):', min_value=int(year2)+1)
            time_gap = int(year2) - int(year1)
            future_steps = (future_year - int(year2)) // time_gap

            if future_steps <= 0:
                st.warning('⚠️ Future year must be after current!')
            else:
                change_matrix = np.zeros((8, 8))
                for from_class in range(8):
                    from_mask = (lc1 == from_class)
                    total_from = np.sum(from_mask)
                    if total_from > 0:
                        for to_class in range(8):
                            to_count = np.sum((lc2 == to_class) & from_mask)
                            change_matrix[from_class, to_class] = to_count / total_from

                prob_matrix = np.linalg.matrix_power(change_matrix, future_steps)

                predicted_lulc = np.full_like(lc2, np.nan)
                for i in range(predicted_lulc.shape[0]):
                    for j in range(predicted_lulc.shape[1]):
                        if not np.isnan(lc2[i, j]):
                            current_class = int(lc2[i, j])
                            predicted_lulc[i, j] = np.argmax(prob_matrix[current_class])

                st.subheader(f'🗺️ Predicted LULC Map for {future_year}')
                plot_map(predicted_lulc, f'Predicted LULC Map for {future_year}', transform2, crs2)

                # Download Predicted
                buffer_pred = io.BytesIO()
                with rasterio.open(
                    buffer_pred, 'w', driver='GTiff',
                    height=predicted_lulc.shape[0], width=predicted_lulc.shape[1],
                    count=1, dtype='float32', crs=profile2['crs'], transform=profile2['transform'],
                    nodata=-9999
                ) as dst:
                    dst.write(predicted_lulc.astype('float32'), 1)
                buffer_pred.seek(0)

                st.download_button(
                    label=f"Download Predicted LULC {future_year} (.tif)",
                    data=buffer_pred,
                    file_name=f'predicted_lulc_{future_year}.tif',
                    mime='image/tiff'
                )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

else:
    st.info('Please upload both TIFF files to continue.')

# Footer
st.markdown("---")
st.caption("Developed by [Your Name] | Powered by Streamlit 🚀")
