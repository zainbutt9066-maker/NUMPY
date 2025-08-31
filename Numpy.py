import streamlit as st
import numpy as np
from scipy import stats


def initialize_session_state():
    """Initialize all session state variables safely"""
    if hasattr(st, 'session_state'):
        if 'elements' not in st.session_state:
            st.session_state.elements = [0.0] * 5
        if 'show_power_section' not in st.session_state:
            st.session_state.show_power_section = False


def main():
    # Initialize session state first
    initialize_session_state()

    st.set_page_config(
        page_title="Numpy Array Builder",
        page_icon="🔢",
        layout="wide"
    )

    st.title("🔢 Numpy Array Builder")
    st.markdown("Create and manage numpy arrays with comprehensive operations!")

    # Sidebar for array configuration
    with st.sidebar:
        st.header("Array Configuration")

        # Array name input
        array_name = st.text_input(
            "Array Name",
            value="my_array",
            help="Enter a name for your array"
        )

        # Array size input
        array_size = st.number_input(
            "Array Size",
            min_value=1,
            max_value=50,
            value=5,
            help="Enter the number of elements (1-50)"
        )

        # Array type selection
        array_dtype = st.selectbox(
            "Data Type",
            options=["int", "float"],
            index=1,
            help="Select the data type for your array"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 Enter Array Elements")

        # Adjust elements list size if array_size changed
        if hasattr(st, 'session_state') and hasattr(st.session_state, 'elements'):
            if len(st.session_state.elements) != array_size:
                if array_size > len(st.session_state.elements):
                    # Extend the list
                    st.session_state.elements.extend([0.0] * (array_size - len(st.session_state.elements)))
                else:
                    # Truncate the list
                    st.session_state.elements = st.session_state.elements[:array_size]

        # Create input fields for each element
        elements = []
        for i in range(array_size):
            if hasattr(st, 'session_state') and hasattr(st.session_state, 'elements'):
                default_val = st.session_state.elements[i] if i < len(st.session_state.elements) else (
                    0 if array_dtype == "int" else 0.0)
            else:
                default_val = 0 if array_dtype == "int" else 0.0

            if array_dtype == "int":
                element = st.number_input(
                    f"Element {i + 1}",
                    value=int(default_val),
                    step=1,
                    key=f"element_{i}"
                )
            else:  # float
                element = st.number_input(
                    f"Element {i + 1}",
                    value=float(default_val),
                    step=0.1,
                    format="%.2f",
                    key=f"element_{i}"
                )

            elements.append(element)

        # Update session state if available
        if hasattr(st, 'session_state'):
            st.session_state.elements = elements

    with col2:
        st.subheader("📊 Array Information")

        try:
            # Create numpy array
            if array_dtype == "int":
                numpy_array = np.array(elements, dtype=int)
            else:
                numpy_array = np.array(elements, dtype=float)

            # Display array information
            st.success(f"✅ Array '{array_name}' created successfully!")

            # Array display
            st.write("**Array Contents:**")
            st.code(f"{array_name} = {numpy_array}", language="python")

            # Array properties
            st.write("**Array Properties:**")
            properties = {
                "Shape": numpy_array.shape,
                "Size": numpy_array.size,
                "Data Type": numpy_array.dtype,
                "Dimensions": numpy_array.ndim,
                "Memory Usage": f"{numpy_array.nbytes} bytes"
            }

            for prop, value in properties.items():
                st.write(f"• **{prop}:** {value}")

        except Exception as e:
            st.error(f"Error creating array: {str(e)}")
            return

    # Statistical Operations Section
    st.markdown("---")
    st.header("📈 Statistical Operations")

    # Create columns for operation buttons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("📊 Calculate Average", use_container_width=True, key="calc_avg"):
            result = np.mean(numpy_array)
            st.success(f"**Average:** {result:.4f}")
            st.code(f"np.mean({array_name}) = {result:.4f}")

    with col2:
        if st.button("➕ Calculate Sum", use_container_width=True, key="calc_sum"):
            result = np.sum(numpy_array)
            st.success(f"**Sum:** {result:.4f}")
            st.code(f"np.sum({array_name}) = {result:.4f}")

    with col3:
        if st.button("📏 Calculate Variance", use_container_width=True, key="calc_var"):
            result = np.var(numpy_array)
            st.success(f"**Variance:** {result:.4f}")
            st.code(f"np.var({array_name}) = {result:.4f}")

    with col4:
        if st.button("📈 Calculate Maximum", use_container_width=True, key="calc_max"):
            result = np.max(numpy_array)
            st.success(f"**Maximum:** {result:.4f}")
            st.code(f"np.max({array_name}) = {result:.4f}")

    with col5:
        if st.button("📉 Calculate Minimum", use_container_width=True, key="calc_min"):
            result = np.min(numpy_array)
            st.success(f"**Minimum:** {result:.4f}")
            st.code(f"np.min({array_name}) = {result:.4f}")

    # Second row of buttons
    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        if st.button("🎯 Calculate Median", use_container_width=True, key="calc_median"):
            result = np.median(numpy_array)
            st.success(f"**Median:** {result:.4f}")
            st.code(f"np.median({array_name}) = {result:.4f}")

    with col7:
        if st.button("📊 Standard Deviation", use_container_width=True, key="calc_std"):
            result = np.std(numpy_array)
            st.success(f"**Standard Deviation:** {result:.4f}")
            st.code(f"np.std({array_name}) = {result:.4f}")

    with col8:
        if st.button("🎵 Harmonic Mean", use_container_width=True, key="calc_harmonic"):
            try:
                if np.all(numpy_array > 0):
                    result = stats.hmean(numpy_array)
                    st.success(f"**Harmonic Mean:** {result:.4f}")
                    st.code(f"scipy.stats.hmean({array_name}) = {result:.4f}")
                else:
                    st.error("Harmonic mean requires positive values only!")
            except Exception as e:
                st.error("Harmonic mean requires positive values only!")

    with col9:
        # Percentile calculation with input
        percentile_value = st.number_input("Percentile (%)", min_value=0, max_value=100, value=50, step=1,
                                           key="percentile_input")
        if st.button("📊 Calculate Percentile", use_container_width=True, key="calc_percentile"):
            result = np.percentile(numpy_array, percentile_value)
            st.success(f"**{percentile_value}th Percentile:** {result:.4f}")
            st.code(f"np.percentile({array_name}, {percentile_value}) = {result:.4f}")

    with col10:
        # Power operation with second array
        if st.button("⚡ Power Operation", use_container_width=True, key="show_power_info"):
            st.info("Use the toggle below to show Power Operation Section")

    # Power Operation Section - Use checkbox for better compatibility
    st.markdown("---")
    show_power = st.checkbox("⚡ Show Power Operation Section", key="power_section_toggle")

    if show_power:
        st.subheader("⚡ Power Operation: Array1 ^ Array2")

        col_power1, col_power2 = st.columns(2)

        with col_power1:
            st.write("**Base Array (Current Array):**")
            st.code(f"Base = {numpy_array}")

        with col_power2:
            st.write("**Exponent Array (Enter values):**")
            power_elements = []
            for i in range(array_size):
                power_element = st.number_input(
                    f"Exponent {i + 1}",
                    value=2.0,
                    step=0.1,
                    key=f"power_element_{i}"
                )
                power_elements.append(power_element)

            power_array = np.array(power_elements)
            st.code(f"Exponent = {power_array}")

        if st.button("🚀 Calculate Power", use_container_width=True, key="calc_power"):
            try:
                result = np.power(numpy_array, power_array)
                st.success("**Power Operation Result:**")
                st.code(f"np.power({array_name}, exponent_array) = {result}")

                # Show element-wise calculation
                st.write("**Element-wise calculation:**")
                for i in range(len(numpy_array)):
                    st.write(f"• {numpy_array[i]}^{power_array[i]} = {result[i]:.4f}")
            except Exception as e:
                st.error(f"Error in power operation: {str(e)}")

    # All Operations Summary
    st.markdown("---")
    st.subheader("📋 All Operations Summary")

    if st.button("🔄 Calculate All Statistics", use_container_width=True, key="calc_all"):
        try:
            st.write("**Complete Statistical Analysis:**")

            stats_dict = {
                "Average (Mean)": np.mean(numpy_array),
                "Sum": np.sum(numpy_array),
                "Variance": np.var(numpy_array),
                "Standard Deviation": np.std(numpy_array),
                "Maximum": np.max(numpy_array),
                "Minimum": np.min(numpy_array),
                "Median": np.median(numpy_array),
                "25th Percentile": np.percentile(numpy_array, 25),
                "75th Percentile": np.percentile(numpy_array, 75),
                "90th Percentile": np.percentile(numpy_array, 90),
            }

            # Add harmonic mean if all values are positive
            if np.all(numpy_array > 0):
                stats_dict["Harmonic Mean"] = stats.hmean(numpy_array)

            # Display in a nice format
            for stat_name, stat_value in stats_dict.items():
                st.write(f"• **{stat_name}:** {stat_value:.4f}")

            # Generate complete Python code
            python_code = f"""import numpy as np
from scipy import stats

# Array definition
{array_name} = np.array({list(elements)})

# Statistical calculations
average = np.mean({array_name})
total_sum = np.sum({array_name})
variance = np.var({array_name})
std_dev = np.std({array_name})
maximum = np.max({array_name})
minimum = np.min({array_name})
median = np.median({array_name})
percentile_25 = np.percentile({array_name}, 25)
percentile_75 = np.percentile({array_name}, 75)
percentile_90 = np.percentile({array_name}, 90)

print(f"Average: {{average:.4f}}")
print(f"Sum: {{total_sum:.4f}}")
print(f"Variance: {{variance:.4f}}")
print(f"Standard Deviation: {{std_dev:.4f}}")
print(f"Maximum: {{maximum:.4f}}")
print(f"Minimum: {{minimum:.4f}}")
print(f"Median: {{median:.4f}}")
print(f"25th Percentile: {{percentile_25:.4f}}")
print(f"75th Percentile: {{percentile_75:.4f}}")
print(f"90th Percentile: {{percentile_90:.4f}}")

# Harmonic mean (for positive values only)
if np.all({array_name} > 0):
    harmonic_mean = stats.hmean({array_name})
    print(f"Harmonic Mean: {{harmonic_mean:.4f}}")
"""

            st.code(python_code, language="python")

            # Download button for complete code
            st.download_button(
                label="📥 Download Complete Analysis Code",
                data=python_code,
                file_name=f"{array_name}_complete_analysis.py",
                mime="text/plain",
                key="download_code"
            )

        except Exception as e:
            st.error(f"Error in complete analysis: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Made by Zain Butt by using Streamlit, NumPy, and SciPy</p>
            <p>Complete statistical analysis and array operations toolkit!</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()