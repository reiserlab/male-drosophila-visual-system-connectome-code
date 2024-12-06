# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: default
#     language: python
#     name: python3
# ---

# %%
import shutil
from pathlib import Path
import sys
import os
import fitz
from dotenv import load_dotenv, find_dotenv
load_dotenv()
PROJECT_ROOT = Path(find_dotenv()).parent
sys.path.append(str(PROJECT_ROOT.joinpath("src")))
print(f"Project root directory: {PROJECT_ROOT}")


# %% [markdown]
# Helper notebook to generate the Catalog from single pages.

# %%
doc_out = fitz.open()

page_counter = 1 # 1-based for bookmarks / toc

page_toc = []

df_dict = {
    "creator": "Reiser Lab"
  , "producer": "Janelia Research Campus"
  , "creationDate": fitz.get_pdf_now()
  , "modDate": fitz.get_pdf_now()
  , "title": "Cell Type Catalog, Supplementary Figure 1 to 'Connectome-driven neural inventory of a complete visual system'"
  , "author": "Aljoscha Nern, Frank Loesche, Shin-ya Takemura, Laura E Burnett, Marisa Dreher, Eyal Gruntman, Judith Hoeller, "\
        "Gary B Huang, Michał Januszewski, Nathan C Klapoetke, Sanna Koskela, Kit D Longden, Zhiyuan Lu, Stephan Preibisch, "\
        "Wei Qiu, Edward M Rogers, Pavithraa Seenivasan, Arthur Zhao, John Bogovic, Brandon S Canino, Jody Clements, Michael Cook, "\
        "Samantha Finley-May, Miriam A Flynn, Imran Hameed, Alexandra MC Fragniere, Kenneth J Hayworth, Gary Patrick Hopkins, "\
        "Philip M Hubbard, William T Katz, Julie Kovalyak, Shirley A Lauchie, Meghan Leonard, Alanna Lohff, Charli A Maldonado, "\
        "Caroline Mooney, Nneoma Okeoma, Donald J Olbris, Christopher Ordish, Tyler Paterson, Emily M Phillips, Tobias Pietzsch, "\
        "Jennifer Rivas Salinas, Patricia K Rivlin, Philipp Schlegel, Ashley L Scott, Louis A Scuderi, Satoko Takemura, Iris Talebi, "\
        "Alexander Thomson, Eric T Trautman, Lowell Umayam, Claire Walsh, John J Walsh, C Shan Xu, Emily A Yakal, Tansy Yang, "\
        "Ting Zhao, Jan Funke, Reed George, Harald F Hess, Gregory SXE Jefferis, Christopher Knecht, Wyatt Korff, Stephen M Plaza, "\
        "Sandro Romani, Stephan Saalfeld, Louis K Scheffer, Stuart Berg, Gerald M Rubin, Michael B Reiser"
  , "subject": "Connectome-driven neural inventory of a complete visual system"
  , "keywords": "connectome, Drosophila, Drosophila melanogaster, vision"
}

doc_out.set_metadata(df_dict)

has_typst = shutil.which("typst")
if has_typst:
    os.system('typst compile CellTypeCatalog_Title.typ 2>/dev/null') 
else:
    print("Using pre-generated Title page")

title_doc =  "CellTypeCatalog_Title.pdf"
tmp_title = fitz.open(title_doc)
doc_out.insert_pdf(tmp_title)
page_counter += 1

for idx in range(0, 35):
    for fname in ["Gallery_Group-", "Summary_Group-"]:
        tmp_fn = PROJECT_ROOT / "results" / "fig_summary" / f"{fname}{idx:02d}.pdf"
        tmp_doc = fitz.open(tmp_fn)
        doc_out.insert_pdf(tmp_doc)
        tmp_toc = tmp_doc.get_toc(simple=False)
        for toc_entry in tmp_toc:
            toc_e = toc_entry
            toc_e[2] = page_counter
            page_toc.append(toc_e)
        page_counter += 1

doc_out.set_toc(page_toc)


catalog_fn = PROJECT_ROOT / "results" / "fig_summary" / "Cell Type Catalog.pdf"
doc_out.save(catalog_fn)

has_ocrmypdf = shutil.which("ocrmypdf")
if has_ocrmypdf:
    comp_cat_fn = catalog_fn.parent / "Cell Type Catalog optimzed.pdf"
    os.system(f"ocrmypdf --tesseract-timeout=0 --optimize 1 --skip-text '{catalog_fn}' '{comp_cat_fn}'")


# %%
