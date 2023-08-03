import streamlit as st
import svm
import datasets
import plots
import random
import sklearn

RANDOM_SEED = 147
LEARNING_RATE_VALUES = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
X_train = None
X_test = None
y_train = None
y_test = None

train_acc = None
test_acc = None
fig_class = None
ax_class = None
fig_learn = None
ax_learn = None
def generate_new_seed():
    global RANDOM_SEED
    RANDOM_SEED = random.randint(0, 1000)


def get_dataset(data_label, scale=True, random_seed=0):
    global X_train, X_test, y_train, y_test
    if data_label == DATASETS[0]:
        X_train, X_test, y_train, y_test = datasets.get_blobs(
            n_samples=n_samples,
            cluster_std=cluster_std,
            random_seed=random_seed
        )
    elif data_label == DATASETS[1]:
        X_train, X_test, y_train, y_test = datasets.get_circles(
            n_samples=n_samples,
            noise=noise,
            random_seed=random_seed
        )
    if scale:
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)


def classify_svm():
    global train_acc, test_acc, fig_class, ax_class, fig_learn, ax_learn
    kernel = svm.Kernel(kernel_type=selected_kernel)
    classifier = svm.SVM(kernel=kernel, accuracy=margin_penalty)
    classifier.fit(data=X_train, targets=y_train, n_iterations=num_of_iterations, learning_rate=learning_rate)
    train_acc = classifier.calculate_score(X_train, y_train)
    test_acc = classifier.calculate_score(X_test, y_test)
    x_ticks, y_ticks, grid_classification = classifier.classification_bound_2d()
    fig_class, ax_class = plots.plot_svm_classification(
        X_train,
        y_train,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        classification_grid=grid_classification.T,
        support_vectors=classifier.svcs_idx
    )
    fig_learn, ax_learn = plots.plot_learning_curve(classifier.losses)


DATASETS = ("blobs", "circles")
KERNELS = ("linear", "sigmoid", "rbf")

st.title("SVM CLASSIFIER")
st.subheader("""
Explore svm classifier possibilities on example datasets
""")
st.divider()
st.sidebar.write("DATA AND KERNELS")
selected_data = st.sidebar.selectbox("Select dataset", DATASETS)
st.sidebar.button("RANDOMIZE SEED", on_click=generate_new_seed())
st.sidebar.write(f"CURRENT SEED: :blue[{str(RANDOM_SEED)}]")

st.sidebar.divider()
st.sidebar.write("DATASET SETTINGS")
if selected_data == DATASETS[0]:
    n_samples = st.sidebar.slider(label="Number of samples", min_value=20, max_value=500, value=200)
    cluster_std = st.sidebar.slider(label="Variance", min_value=0.0, max_value=5.0, value=0.5)

elif selected_data == DATASETS[1]:
    n_samples = st.sidebar.slider(label="Number of samples", min_value=20, max_value=500, value=200)
    noise = st.sidebar.slider(label="Noise", min_value=0.0, max_value=1.0, value=0.2)
get_dataset(selected_data, scale=True, random_seed=RANDOM_SEED)

st.sidebar.divider()
selected_kernel = st.sidebar.selectbox("Select SVM kernel", KERNELS)
st.sidebar.write(selected_kernel.upper() + " KERNEL SETTINGS")
if selected_kernel == KERNELS[0]:
    linear_bias = st.sidebar.slider(label="Linear bias", min_value=0.0, max_value=10.0, value=0.0)
elif selected_kernel == KERNELS[1]:
    sigmoid_gamma = st.sidebar.slider(label="Sigmoid gamma", min_value=0.0, max_value=5.0, value=.1)
    sigmoid_bias = st.sidebar.slider(label="Sigmoid bias", min_value=0.0, max_value=10.0, value=1.0)
elif selected_kernel == KERNELS[2]:
    use_custom_gamma = st.sidebar.checkbox("Use custom gamma value", value=False)
    if use_custom_gamma:
        rbf_gamma = st.sidebar.slider(label="Sigmoid bias", min_value=0.0, max_value=10.0, value=1.0)
    else:
        rbf_gamma = None

st.sidebar.divider()
st.sidebar.write("SVM OPTIMIZER SETTINGS")
num_of_iterations = st.sidebar.slider(label="Number of iterations", min_value=5, max_value=2000, value=200)
margin_penalty = st.sidebar.slider(label="Margin penalty", min_value=0., max_value=10., value=1.0)
learning_rate = st.sidebar.select_slider(label="Learning rate", options=LEARNING_RATE_VALUES)
#st.sidebar.button(label="PERFORM CLASSIFICATION", on_click=classify_svm())
classify_svm()

st.write(f"Data type: **{selected_data.upper()}**")
st.write(f"Kernel type: **{selected_kernel.upper()}**")

col_l, col_r = st.columns([3,1])
col_r.write(f"Train accuracy: **{train_acc * 100:.2f}%**")
col_r.write(f"Test accuracy: **{test_acc * 100:.2f}%**")

col_l.pyplot(fig_class)
st.divider()
col_l, col_r = st.columns([3,1])
col_l.pyplot(fig_learn)
