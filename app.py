import streamlit as st
import pandas as pd
from team_formation_tool import Optimizer
import io

# Set page configuration
st.set_page_config(
    page_title="Team Formation Tool",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("Team Formation Tool")
st.write("Upload Excel file, view its data, and perform optimization.  Created by John Mathieu and David Bergman, University of Connecitcut.")

# Initialize session state for logs
if "full_logs" not in st.session_state:
    st.session_state.full_logs = []  # Persist logs across interactions

# --- SECTION 1: File Upload ---
st.header("📁 File Upload")
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])

if uploaded_file:
    # Read the Excel file and get all sheet names
    excel_data = pd.ExcelFile(uploaded_file)
    sheet_names = excel_data.sheet_names  # Get all sheet names

    st.write("### View Sheets")
    
    # Iterate through each sheet and display it in an expandable section
    for sheet in sheet_names:
        with st.expander(f"Sheet: {sheet}", expanded=False):
            # Read and display the sheet data
            sheet_data = pd.read_excel(excel_data, sheet_name=sheet)
            st.dataframe(sheet_data)
else:
    st.info("Please upload an Excel file to get started.")

# --- SECTION 2: Optimization ---
st.header("⚙️ Optimization")

if uploaded_file:
    # User input: Time allowed for optimization (in seconds)
    st.write("### Set Optimization Parameters")
    with st.container():
        col1, col2 = st.columns([1, 4])  # Create two columns, first is smaller
        with col1:
            time_limit = st.number_input(
                "Time (sec):",
                min_value=1,  # Minimum value is 1 second
                max_value=3600,  # Maximum value is 3600 seconds (1 hour)
                value=60,  # Default value
                step=1,  # Increment by 1 second
                help="Enter the maximum time allowed for the optimization in seconds."
            )
    with st.container():
        col1, col2 = st.columns([1, 4])  # Create two columns, first is smaller
        with col1:
            n_teams = st.number_input(
                "Number of Desired Teams: ",
                min_value=1,
                max_value=40,
                value=2,
                step=1,
                help="Enter the number of desired teams."
            )

    # Show the "Optimize" button
    if st.button("Optimize"):
        st.write("### Optimization Process")
        st.write(f"Running optimization with a time limit of {time_limit} seconds...")

        # Instantiate the optimizer and pass the time limit
        optimizer = Optimizer(excel_data, time_limit=time_limit, n_teams=n_teams)
        optimizer.prep_model()

        # Add a styled container for the logs
        st.markdown(
            """
            <style>
            .log-container {
                background-color: #f9f9f9;
                padding: 15px;
                border: 1px solid #ccc;
                border-radius: 5px;
                height: 300px;
                overflow-y: auto;
                font-family: monospace;
                font-size: 14px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        log_display = st.empty()

        # Run the model. We expect run_model() to yield logs AND return a final DataFrame
        # You can structure run_model so it yields logs in real-time, but also returns
        # a final DataFrame at the end. For simplicity, let's assume run_model() just
        # returns the DataFrame after it has yielded all logs:
        results_df = None

        # If run_model() is a generator that yields logs, we can do something like:
        gen = optimizer.run_model()
        while True:
            try:
                log = next(gen)
                if isinstance(log, str):
                    # It's a log message
                    clean_log = log.strip()
                    st.session_state.full_logs.append(clean_log)
                    # Display only the latest logs (emulate auto-scroll)
                    log_display.markdown(
                        f"<div class='log-container'>{'<br>'.join(st.session_state.full_logs[-100:])}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    # If we get a dataframe at some point
                    if isinstance(log, pd.DataFrame):
                        results_df = log
            except StopIteration as e:
                # The generator is done. If it returns a value (DataFrame) at the end:
                if e.value is not None and isinstance(e.value, pd.DataFrame):
                    results_df = e.value
                break

        st.success("Optimization Complete!")
        st.write("### Results")

        # If we didn't get the DataFrame as a yield but rather a return, we have it in results_df
        # If none is returned, you might need to adapt your run_model logic
        if results_df is not None:
            st.dataframe(results_df)

            # Prepare CSV for download
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            # Download button
            st.download_button(
                label="Download Results as CSV",
                data=csv_data,
                file_name="optimized_results.csv",
                mime="text/csv"
            )
        else:
            st.write("No results to display yet. Make sure `run_model()` returns or yields a DataFrame.")

        # Generate and display the plot
        st.write("### Plot of Optimization Results")
        plot_buffer = optimizer.create_plot()  # Call the plot creation method
        st.image(plot_buffer,width=200)

        # You can also display other JSON results or final parameters
        # st.json({
        #     "optimized_value_1": optimizer.param1,
        #     "optimized_value_2": optimizer.param2,
        #     "status": "Success"
        # })
else:
    st.warning("Upload an Excel file to enable the optimization process.")
