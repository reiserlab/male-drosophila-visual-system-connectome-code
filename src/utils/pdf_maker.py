from abc import ABC
from pathlib import Path
import re

from dotenv import find_dotenv

import fitz


class PDFMaker(ABC):

    """
    Interface for the MuPDF library
    """

    def __init__(
        self
      , width:float
      , height:float
      , resolution:float
      , margin:tuple=None
    ):
        """
        Initialize the object with width and height of the PDF to generate

        Parameters
        ----------
        width : float
            width in mm
        height : float
            height in mm
        resolution : float
            resolution in dpi (not really used?)
        margin : tuple, default=None
            margin of the page in mm, [margin_x, margin_y]. `None` defaults to [0, 0]
        """

        assert width > 0, f"width must be > 0, not {width}"
        assert height > 0, f"height muyst be > 0, not {height}"
        assert resolution > 0, f"resolution must be > 0, not {resolution}"

        if margin is None:
            margin = (0, 0)

        assert len(margin) == 2,\
            "gotta have 2 values for margin"

        conversion = 72 / 25.4  # 72 pixel per inch, 25.4 mm per inch

        self.__rect = fitz.Rect(0, 0, width * conversion, height * conversion)
        self.__resolution = resolution
        self.__pdf = fitz.open()
        self.__page = self.__pdf.new_page(width=self.__rect.x1, height=self.__rect.y1)
        self.__margin = [margin[0] * conversion, margin[1] * conversion]


    def get_page_layout_rows_cols(
        self
      , rows:int
      , cols:int
      , aspect_ratio:float=None
    ):
        """
        Given the number of rows / columns to arrange plots into, determine the
        layout of the images and the size they should be.

        Parameters
        ----------
        rows : int
            number of rows
        cols : int
            number of cols
        aspect_ratio: float, default = None
            aspect ratio of plots to use.
            If 'None' will distort the images to fill the pdf.

        Returns
        ----------
        coords : list
            A list of tuples containing the coordinates of each image.
        img_width : int
            width of image to add to pdf
        img_height : int
            height of image to add to pdf
        """
        margin_x, margin_y = self.__margin
        paper_width = self.__rect.x1 - (2*margin_x)
        paper_height = self.__rect.y1 - (2*margin_y)

        if aspect_ratio is None:
            img_width = paper_width / cols
            img_height = paper_height / rows
        else:
            img_width = (paper_height * aspect_ratio) / cols
            img_height = img_width / aspect_ratio

        if img_width * cols > paper_width:
            img_width = paper_width / cols
            img_height = img_width / aspect_ratio
        if img_height * rows > paper_height:
            img_height = paper_height / rows
            img_width = img_height * aspect_ratio

        # Initialize lists to store coordinates
        coords = []

        # Iterate over each row
        for i in range(rows):
            # Iterate over each column
            for j in range(cols):
                # Calculate the top-left and bottom-right coordinates of the image
                top_left_x = (j * img_width) + margin_x
                top_left_y = (i * img_height) + margin_y
                bottom_right_x = ((j + 1) * img_width) + margin_x
                bottom_right_y = ((i + 1) * img_height) + margin_y

                # Append coordinates to the list
                coords.append((top_left_x, top_left_y, bottom_right_x, bottom_right_y))

        return coords, img_width, img_height


    def add_image(
        self
      , filename:str|Path
      , coord:list
    ) -> None:
        """
        Add the image previously generated by blender to the PDF file.

        Parameters
        ----------
        filename : str | Path
            location of the image file.
        coord : list
            list of coordinates of the image's position.
        """
        if isinstance(filename, str):
            filename = Path(filename)
        # TODO(@floesche): check if file exists etc…
        img = fitz.open(filename)
        # TODO(@floesche): if resolution isn't working well, use the zooming function:
        # zoomx = self.__rect.x1 / img[0].rect.x1
        # zoomy = self.__rect.y1 / img[0].rect.y1
        # zoom = min(zoomx, zoomy)
        # mat = fitz.Matrix(zoom, zoom)
        px = img[0].get_pixmap(alpha=True, dpi=self.__resolution)
        # reduce image size
        px.shrink(1)
        # FIX ME - check why this doesn't seem to change the file size
        # px.set_dpi(self.__resolution, self.__resolution)
        # self.__page.insert_image(self.__rect, pixmap=px)
        self.__page.insert_image(fitz.Rect(coord), pixmap=px, keep_proportion=True)


    def add_text(
        self
      , text:str
      , position:list
      , color:list
      , align:str="l"
      , font_size:int=5
    ) -> None:
        """
        Add text to the PDF file

        Parameters
        ----------
        text : str
            The text to add
        position : list
            [x, y] position of the text.
        color : list
            [r, g, b] color of the text (0 <= values <= 1)
        align : str
            alignment of text to the left ('l'), right ('r') or centre ('c')
        font_size : int
            font size of title text in pt
        """
        assert align in ['l', 'c', 'r'], \
            f"can only align 'l'eft, 'c'enter, or 'r'ight, not {align}"
        if 0<=position[0]<=1 and 0<=position[1]<=1:
            posx = self.__rect.x1 * position[0]
            posy = self.__rect.y1 * (1-position[1])
        else:
            posx, posy = position
        col = color[:3]
        fontfile = str(Path(find_dotenv()).parent / "cache" / "layout" / "Arial.ttf")
        font = fitz.Font(fontfile=fontfile)
        nmbr = None
        side = None

        # regular expression patterns to extract number and side from the text
        num_side_pattern = r"(.+)\s\((\w+)\)\s\((\d+)\)"
        num_pattern = r"(.+)\s\((\d+)\)"
        side_pattern = r"(.+)\s\((\w+)\)"

        # check type and side and number
        mtch = re.search(num_side_pattern, text)
        if mtch:
            text, side, nmbr = mtch.groups()
        else:
            # check type and number
            mtch = re.search(num_pattern, text)
            # check type and side
            mtch2 = re.search(side_pattern, text)
            if mtch and mtch.group(2):
                nmbr = mtch.group(2)
                text = mtch.group(1)
            elif mtch2 and mtch2.group(2):
                side = mtch2.group(2)
                text = mtch2.group(1)

        if align != "l":
            shadow = fitz.TextWriter(self.__page.rect, color=col)
            _, bottom_r = shadow.append(
                pos=(posx, posy)
              , text=text
              , font=font
              , fontsize=font_size
            )
            if align == "r":
                posx -= bottom_r[0] - posx
            if align == "c":
                posx -= (bottom_r[0] - posx) / 2

        tw = fitz.TextWriter(self.__page.rect, color=col)
        _, tw_r = tw.append(
            pos=(posx, posy)
          , text=text
          , font=font
          , fontsize=font_size
        )
        tw.write_text(self.__page)

        if side and nmbr is None:
            tw_side = fitz.TextWriter(self.__page.rect, color=[0.6, 0.6, 0.6])
            tw_side.append(
                pos=tw_r
              , text=f" ({side})"
              , font=font
              , fontsize=font_size - 1
              , small_caps=True
            )
            tw_side.write_text(self.__page)
        if nmbr and side is None:
            tw_num = fitz.TextWriter(self.__page.rect, color=[0.6, 0.6, 0.6])
            tw_num.append(
                pos=tw_r
              , text=f" {nmbr}"
              , font=font
              , fontsize=font_size - 1
              , small_caps=True
            )
            tw_num.write_text(self.__page)
        if nmbr and side:
            tw_side = fitz.TextWriter(self.__page.rect, color=[0.6, 0.6, 0.6])
            _, tw_side_r = tw_side.append(
                pos=tw_r
              , text=f" ({side})"
              , font=font
              , fontsize=font_size - 1
              , small_caps=True
            )
            tw_side.write_text(self.__page)
            tw_num = fitz.TextWriter(self.__page.rect, color=[0.6, 0.6, 0.6])
            tw_num.append(
                tw_side_r
              , f" {nmbr}"
              , font=font
              , fontsize=font_size - 1
              , small_caps=True
            )
            tw_num.write_text(self.__page)


    def save(
        self
      , filename:str
      , directory:str=None
    ):
        """
        Parameters
        ----------
        filename : str
            filename to save the PDF to
        dirctory : str, default=None
            If another directory than `PROJECT_ROOT / results / gallery` is requried
        """
        if directory is None:
            project_root = Path(find_dotenv()).parent
            directory = project_root / "results" / "gallery"
        else:
            directory = Path(directory)

        self.__pdf.save(
            directory / filename
          , garbage=4
          , clean=True
          , deflate=True
          , deflate_images=True
        )