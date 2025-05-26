# Sample data: list of tuples (token, class)
tokens = [
    ("Hello", 1),
    ("world", 2),
    (",", 3),
    ("this", 1),
    ("is", 2),
    ("a", 3),
    ("test", 1),
    (".", 2)
]

color_palette = ["#4285F4", "#FFAB40", "#A64D79", "#97D077"]

# Mapping from class to a background color.
class_colors = {
    0: color_palette[0], 
    1: color_palette[1],
    2: color_palette[2],
    3: color_palette[3],
}

def generate_html(prompt, tokens):
    # Start building the HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Token Visualization</title>
        <style>
            /* Optional styling for smoother visualization */
            .token {
                padding: 2px 4px;
                margin: 1px;
                border-radius: 3px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <p>
    """

    # Add the prompt to the HTML content
    html_content += f'<p class="token" style="background-color: #f0f0f0;"><strong>Prompt:</strong> {prompt}</p> '

    # Generate a span for each token with the corresponding background color.
    for layer_idx, layer_tokens in enumerate(tokens):
        if layer_idx == len(tokens) - 1:
            html_content += f'<p><strong>Majority Vote:</strong> '
        else:
            html_content += f'<p><strong>Layer {layer_idx+1}:</strong> '
        for token, cls in layer_tokens:
            color = class_colors.get(cls, "#ffffff")  # default to white if class is missing
            html_content += f'<span class="token" style="background-color: {color};">{token}</span> '
        html_content += "</p>"

        html_content += """
        </p>
    </body>
    </html>
    """

    # Write the HTML content to a file
    with open("outputs/output.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("HTML file generated: output.html")
