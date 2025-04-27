# Imports
import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import io
from pyproj import Transformer
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Title
st.title('🌎 Land Use Land Cover (LULC) Change Detection and Future Prediction App')

# Class Labels and Colors (0-8)
class_label_mapping = {
    0: 'Water', 1: 'Trees', 2: 'Grass', 3: 'Flooded Vegetation',
    4: 'Crops', 5: 'Shrub and Scrub', 6: 'Built Area', 7: 'Bare Ground',
    8: 'Snow and Ice'
}
class_color_mapping = {
    0: '#419bdf', 1: '#397d49', 2: '#88b053', 3: '#7a87c6',
    4: '#e49635', 5: '#dfc35a', 6: '#c4281b', 7: '#a59b8f',
    8: '#b39fe1'
}

# User Inputs
st.subheader('🗓️ Specify Years')
year1 = st.text_input("Enter year for first TIFF (e.g., 2010):", value="Year 1")
year2 = st.text_input("Enter year for second TIFF (e.g., 2020):", value="Year 2")

uploaded_file_1 = st.file_uploader(f"Upload TIFF for {year1}", type=["tif", "tiff"])
uploaded_file_2 = st.file_uploader(f"Upload TIFF for {year2}", type=["tif", "tiff"])

# Color Mask Creator
def create_color_mask(image_array, classes_present):
    color_image = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.float32)
    for cls in classes_present:
        mask = (image_array == cls)
        rgb = mcolors.to_rgb(class_color_mapping[cls])
        color_image[mask] = rgb
    return color_image

# Plotting Function
def plot_map(image_array, title, transform, crs, classes_present, is_transition=False, transition_info=None):
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
        cmap = mcolors.ListedColormap(transition_info['colors'])
        ax.imshow(transition_info['mapped_array'], cmap=cmap, interpolation='nearest')
        labels = transition_info['labels']
        colors = transition_info['colors']
    else:
        color_image = create_color_mask(image_array, classes_present)
        ax.imshow(color_image, interpolation='nearest')
        labels = [class_label_mapping[c] for c in classes_present]
        colors = [class_color_mapping[c] for c in classes_present]

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

# Main App
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
            st.error('❌ Error: Files have different dimensions!')
        else:
            # Mask both negative and huge positive invalid values
            lc1 = np.where((lc1 <= -9999) | (lc1 >= 2147483647), np.nan, lc1)
            lc2 = np.where((lc2 <= -9999) | (lc2 >= 2147483647), np.nan, lc2)

            valid_mask = (~np.isnan(lc1)) & (~np.isnan(lc2))

            # Plot Year1
            existing_classes_lc1 = sorted(list(set(np.unique(lc1[~np.isnan(lc1)]).astype(int))))
            st.subheader(f'🗺️ LULC Map for {year1}')
            plot_map(lc1, f'LULC Map for {year1}', transform1, crs1, existing_classes_lc1)

            # Plot Year2
            existing_classes_lc2 = sorted(list(set(np.unique(lc2[~np.isnan(lc2)]).astype(int))))
            st.subheader(f'🗺️ LULC Map for {year2}')
            plot_map(lc2, f'LULC Map for {year2}', transform2, crs2, existing_classes_lc2)

            # 📋 Pixel Count Table
            st.subheader('📋 Pixel Count Table (Year 1 vs Year 2)')
            pixel_count_table = pd.DataFrame({
                'Class': list(range(9)),
                'Class Name': [class_label_mapping[i] for i in range(9)],
                f'Pixels ({year1})': [np.nansum(lc1 == i) for i in range(9)],
                f'Pixels ({year2})': [np.nansum(lc2 == i) for i in range(9)]
            })
            st.dataframe(pixel_count_table)

            # 📋 Transition Matrix Table
            st.subheader('📋 Transition Matrix Table')
            transition_matrix = np.zeros((9, 9), dtype=int)
            for i in range(9):
                for j in range(9):
                    transition_matrix[i, j] = np.nansum((lc1 == i) & (lc2 == j))
            transition_df = pd.DataFrame(transition_matrix,
                                         index=[f'From {class_label_mapping[i]}' for i in range(9)],
                                         columns=[f'To {class_label_mapping[i]}' for i in range(9)])
            st.dataframe(transition_df)

            # Transition Map
            transition_map = lc1 * 100 + lc2
            transitions_unique = np.unique(transition_map[valid_mask]).astype(int)

            transition_mapped = np.copy(transition_map)
            trans_idx = {code: idx for idx, code in enumerate(transitions_unique)}
            for code, idx in trans_idx.items():
                transition_mapped[transition_map == code] = idx

            cmap_transitions = cm.get_cmap('tab20', len(transitions_unique))
            transition_colors = [cmap_transitions(i) for i in range(len(transitions_unique))]

            transition_labels = []
            for code in transitions_unique:
                code_str = str(code).zfill(4)
                from_class = int(code_str[:-2])
                to_class = int(code_str[-2:])
                from_label = class_label_mapping.get(from_class, 'Unknown')
                to_label = class_label_mapping.get(to_class, 'Unknown')
                transition_labels.append(f"{from_label} ➔ {to_label}")

            st.subheader(f'🗺️ Transition Map ({year1} ➔ {year2})')
            transition_info = {'mapped_array': transition_mapped, 'colors': transition_colors, 'labels': transition_labels}
            plot_map(transition_map, f'Transition Map ({year1} ➔ {year2})', transform1, crs1,
                     classes_present=[], is_transition=True, transition_info=transition_info)

            # 📥 Download Transition Map
            buffer_transition = io.BytesIO()
            with rasterio.open(
                buffer_transition, 'w', driver='GTiff',
                height=transition_map.shape[0],
                width=transition_map.shape[1],
                count=1, dtype='float32',
                crs=profile1['crs'], transform=profile1['transform'],
                nodata=-9999
            ) as dst:
                dst.write(transition_map.astype('float32'), 1)
            buffer_transition.seek(0)
            st.download_button(
                label=f"Download Transition Map ({year1} ➔ {year2}) (.tif)",
                data=buffer_transition,
                file_name=f'transition_map_{year1}_{year2}.tif',
                mime='image/tiff'
            )

            # 📈 Future LULC Prediction
            st.subheader('📈 Future Land Use Prediction (Markov Chain)')
            future_year = st.number_input(f'Enter future year after {year2}:', min_value=int(year2)+1, step=1)
            time_gap = int(year2) - int(year1)
            future_steps = (future_year - int(year2)) // time_gap

            if future_steps > 0:
                prob_matrix = transition_matrix / np.maximum(transition_matrix.sum(axis=1, keepdims=True), 1)
                prob_matrix = np.linalg.matrix_power(prob_matrix, future_steps)

                predicted_lulc = np.full_like(lc2, np.nan)
                for i in range(predicted_lulc.shape[0]):
                    for j in range(predicted_lulc.shape[1]):
                        if not np.isnan(lc2[i, j]):
                            current_class = int(lc2[i, j])
                            predicted_lulc[i, j] = np.argmax(prob_matrix[current_class])

                st.subheader(f'🗺️ Predicted LULC Map for {future_year}')
                plot_map(predicted_lulc, f'Predicted LULC Map for {future_year}', transform2, crs2,
                         classes_present=list(range(9)))

                # 📋 Predicted LULC Pixel Count Table
                st.subheader('📋 Predicted LULC Pixel Count Table')
                predicted_pixel_table = pd.DataFrame({
                    'Class': list(range(9)),
                    'Class Name': [class_label_mapping[i] for i in range(9)],
                    f'Pixels ({future_year})': [np.nansum(predicted_lulc == i) for i in range(9)]
                })
                st.dataframe(predicted_pixel_table)

                # 📥 Download Predicted LULC
                buffer_pred = io.BytesIO()
                with rasterio.open(
                    buffer_pred, 'w', driver='GTiff',
                    height=predicted_lulc.shape[0],
                    width=predicted_lulc.shape[1],
                    count=1, dtype='float32',
                    crs=profile2['crs'], transform=profile2['transform'],
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
    st.info('ℹ️ Please upload both TIFF files to continue.')

# Footer
st.markdown("---")
st.caption("Developed by Sachchidanand Singh | Powered by Streamlit 🚀")
