import svgwrite

SQUARE_SIZE = 5
# for i in range(192):
#     r = rect((25+(i*SQUARE_SIZE), 25),
#              (30+(i*SQUARE_SIZE), 30),
#              stroke_width=0.25, stroke='#800000', fill='#FF5555')
#     r.add_text(i)

def create_svg_with_text(filename):
    # Create an SVG drawing
    dwg = svgwrite.Drawing(filename, size=('800mm', '400mm'), profile='tiny')

    # Add a rectangles
    # Parameters: insert point (top left corner), size
    for i in range(192):
        rect = dwg.add(dwg.rect(insert=(10+(i*SQUARE_SIZE), 10), size=(10, 10),
                                stroke='#800000',   #svgwrite.rgb(10, 10, 16, '%'),
                                fill='#FF5555'))

    # Add text
    # Parameters: insert point, text
    text = dwg.add(dwg.text('Hello, Inkscape!', insert=(20, 50),
                            fill='black'))

    # Save the drawing
    dwg.save()

if __name__ == "__main__":
    create_svg_with_text('example.svg')