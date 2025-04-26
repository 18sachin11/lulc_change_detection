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

# --- Fancy North Arrow ---
# Place 'N' text and arrow outside map
ax.annotate('N', xy=(0.97, 0.98), xycoords='axes fraction',
            fontsize=16, fontweight='bold', ha='center')
ax.annotate('↑', xy=(0.97, 0.94), xycoords='axes fraction',
            fontsize=20, ha='center')

# --- Create bottom legend ---
labels = [f"{str(code)[:-2]} ➔ {str(code)[-2:]}" for code in transitions_unique]
colors = [cmap(i / len(transitions_unique)) for i in range(len(transitions_unique))]
patches = [plt.plot([], [], marker="s", ms=10, ls="", mec=None, color=colors[i],
                    label="{:}".format(labels[i]))[0] for i in range(len(labels))]

leg = ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.55),
                fancybox=True, shadow=True, ncol=5, title="Transitions")

# --- Add Scale Bar below Legend ---
# Create a new axis under the main plot
fig.subplots_adjust(bottom=0.3)  # Make space at bottom

scalebar_ax = fig.add_axes([0.4, 0.08, 0.2, 0.02])  # [left, bottom, width, height]
scalebar_ax.axis('off')

# Draw scale bar
scalebar_length_deg = 0.1
pixel_width_deg = (np.max(x_deg) - np.min(x_deg)) / ncols

# Calculate how many degrees fit in plot
scalebar_ax.plot([0, 1], [0.5, 0.5], color='black', lw=6)
scalebar_ax.text(0, 0.8, '0°', fontsize=10, va='bottom', ha='center')
scalebar_ax.text(1, 0.8, f'{scalebar_length_deg}°', fontsize=10, va='bottom', ha='center')
scalebar_ax.set_xlim(0, 1)
scalebar_ax.set_ylim(0, 1)

plt.tight_layout()

st.pyplot(fig)
