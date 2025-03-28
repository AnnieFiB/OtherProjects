import os

def generate_autoviz_html_report(folder_name="reports_html", output_filename="AutoViz_Report.html"):
    folder_path = os.path.abspath(folder_name)
    html_path = os.path.join(folder_path, output_filename)

    if not os.path.exists(folder_path):
        print(f"❌ Folder '{folder_path}' does not exist.")
        return None

    svg_files = [f for f in os.listdir(folder_path) if f.endswith(".svg")]
    if not svg_files:
        print("⚠️ No .svg files found in the folder.")
        return None

    html_content = "<html><head><title>AutoViz EDA Report</title></head><body>\n"
    html_content += f"<h1>📊 AutoViz EDA Report</h1>\n"
    html_content += f"<p><b>Generated from folder:</b> {folder_name}</p><hr>\n"

    for filename in sorted(svg_files):
        svg_path = os.path.join(folder_path, filename)
        with open(svg_path, "r", encoding="utf-8") as f:
            svg_content = f.read()
        html_content += f"<div>{svg_content}</div>\n<hr>\n"

    html_content += "</body></html>"

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ AutoViz HTML report saved to: {html_path}")
    return html_path