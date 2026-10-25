"""Strict native OMML builders for the Q2 paper.

Every mathematical run is non-empty, uses Cambria Math, and declares whether
its content is mathematical italic or upright.  Composite structures use true
Office Math nodes instead of flattened Unicode formula strings.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

from docx.oxml import OxmlElement
from docx.oxml.ns import qn


MathNode = object


def math_run(text: str, *, italic: bool = False):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OMML runs must contain non-whitespace text")
    run = OxmlElement("m:r")

    math_properties = OxmlElement("m:rPr")
    style = OxmlElement("m:sty")
    style.set(qn("m:val"), "i" if italic else "p")
    math_properties.append(style)
    run.append(math_properties)

    word_properties = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    for key in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{key}"), "Cambria Math")
    word_properties.append(fonts)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "21")
    word_properties.append(size)
    size_cs = OxmlElement("w:szCs")
    size_cs.set(qn("w:val"), "21")
    word_properties.append(size_cs)
    run.append(word_properties)

    node = OxmlElement("m:t")
    node.text = text
    run.append(node)
    return run


def var(text: str):
    return math_run(text, italic=True)


def roman(text: str):
    return math_run(text, italic=False)


def _append_parts(parent, parts: Iterable):
    for part in parts:
        if isinstance(part, str):
            raise TypeError("formula parts must be explicit var()/roman() OMML runs")
        parent.append(part)
    return parent


def box(tag: str, parts: Iterable):
    return _append_parts(OxmlElement(f"m:{tag}"), parts)


def subscript(base: Iterable, lower: Iterable):
    node = OxmlElement("m:sSub")
    node.append(box("e", base))
    node.append(box("sub", lower))
    return node


def superscript(base: Iterable, upper: Iterable):
    node = OxmlElement("m:sSup")
    node.append(box("e", base))
    node.append(box("sup", upper))
    return node


def subsup(base: Iterable, lower: Iterable, upper: Iterable):
    node = OxmlElement("m:sSubSup")
    node.append(box("e", base))
    node.append(box("sub", lower))
    node.append(box("sup", upper))
    return node


def fraction(numerator: Iterable, denominator: Iterable):
    node = OxmlElement("m:f")
    node.append(box("num", numerator))
    node.append(box("den", denominator))
    return node


def accent(parts: Iterable, *, char: str = "̂"):
    node = OxmlElement("m:acc")
    properties = OxmlElement("m:accPr")
    mark = OxmlElement("m:chr")
    mark.set(qn("m:val"), char)
    properties.append(mark)
    node.append(properties)
    node.append(box("e", parts))
    return node


def bar(parts: Iterable):
    node = OxmlElement("m:bar")
    properties = OxmlElement("m:barPr")
    position = OxmlElement("m:pos")
    position.set(qn("m:val"), "top")
    properties.append(position)
    node.append(properties)
    node.append(box("e", parts))
    return node


def delimiter(parts: Iterable, *, begin: str = "(", end: str = ")"):
    node = OxmlElement("m:d")
    properties = OxmlElement("m:dPr")
    begin_node = OxmlElement("m:begChr")
    begin_node.set(qn("m:val"), begin)
    end_node = OxmlElement("m:endChr")
    end_node.set(qn("m:val"), end)
    properties.extend([begin_node, end_node])
    node.append(properties)
    node.append(box("e", parts))
    return node


def nary(parts: Iterable, lower: Iterable, upper: Iterable, *, char: str = "∑"):
    node = OxmlElement("m:nary")
    properties = OxmlElement("m:naryPr")
    operator = OxmlElement("m:chr")
    operator.set(qn("m:val"), char)
    limit_location = OxmlElement("m:limLoc")
    limit_location.set(qn("m:val"), "undOvr")
    grow = OxmlElement("m:grow")
    grow.set(qn("m:val"), "1")
    properties.extend([operator, limit_location, grow])
    node.append(properties)
    node.append(box("sub", lower))
    node.append(box("sup", upper))
    node.append(box("e", parts))
    return node


def function(name: str, arguments: Iterable):
    node = OxmlElement("m:func")
    node.append(box("fName", [roman(name)]))
    node.append(box("e", [delimiter(arguments)]))
    return node


def _indexed(symbol: str, indices: str):
    return subscript([var(symbol)], [var(indices)])


def _copy(node):
    return copy.deepcopy(node)


def model_relation():
    return [
        _indexed("y", "ij"), roman(" = "), _indexed("f", "j"),
        delimiter([_indexed("x", "i")]), roman(" + "), _indexed("ε", "ij"),
        roman(",   "), var("i"), roman(" = 1,…,250,   "), var("j"), roman(" = 1,…,50"),
    ]


def rank_correlation():
    r_ik = _indexed("R", "ik")
    r_ij = _indexed("R", "ij")
    rbar_k = subscript([bar([var("R")])], [var("k")])
    rbar_j = subscript([bar([var("R")])], [var("j")])
    centered_k = delimiter([r_ik, roman(" − "), rbar_k])
    centered_j = delimiter([r_ij, roman(" − "), rbar_j])
    numerator = nary(
        [_copy(centered_k), _copy(centered_j)],
        [var("i"), roman(" = 1")],
        [var("n")],
    )
    denominator_left = superscript(
        [nary([superscript([_copy(centered_k)], [roman("2")])], [var("i"), roman(" = 1")], [var("n")])],
        [fraction([roman("1")], [roman("2")])],
    )
    denominator_right = superscript(
        [nary([superscript([_copy(centered_j)], [roman("2")])], [var("i"), roman(" = 1")], [var("n")])],
        [fraction([roman("1")], [roman("2")])],
    )
    return [
        _indexed("ρ", "kj"), roman(" = "),
        fraction([numerator], [denominator_left, denominator_right]),
    ]


def graph_elastic_objective():
    beta_hat_j = subscript([accent([var("β")])], [var("j")])
    beta_j = _indexed("β", "j")
    residual_norm = subscript(
        [delimiter([_indexed("y", "j"), roman(" − "), var("X"), _copy(beta_j)], begin="‖", end="‖")],
        [roman("2")],
    )
    l1_norm = subscript([delimiter([var("β")], begin="‖", end="‖")], [roman("1")])
    quadratic = [
        superscript([var("β")], [roman("T")]),
        subscript([var("L")], [var("s")]),
        var("β"),
    ]
    objective = [
        fraction([roman("1")], [roman("2"), var("n")]),
        superscript([residual_norm], [roman("2")]),
        roman(" + "), _indexed("λ", "1"), l1_norm,
        roman(" + "), _indexed("λ", "G"), *quadratic,
    ]
    return [
        beta_hat_j, roman(" = "), subscript([roman("arg min")], [var("β")]),
        delimiter(objective, begin="{", end="}"),
    ]


def conditional_importance():
    delta = superscript([_indexed("Δ", "kj")], [roman("cond")])
    f_hat_j = subscript([accent([var("f")])], [var("j")])
    perturbed = function(
        "MAE",
        [
            _indexed("y", "j"), roman(", "), _copy(f_hat_j),
            delimiter(
                [
                    subscript([var("X")], [roman("−"), var("k")]), roman(", "),
                    subscript([accent([var("x")], char="̃")], [var("k")]),
                ]
            ),
        ],
    )
    baseline = function(
        "MAE",
        [_indexed("y", "j"), roman(", "), _copy(f_hat_j), delimiter([var("X")])],
    )
    return [delta, roman(" = "), perturbed, roman(" − "), baseline]


def rrf_score():
    term = fraction(
        [roman("1")],
        [roman("60 + "), subscript([var("r")], [var("ℓkj")])],
    )
    return [
        _indexed("S", "kj"), roman(" = "),
        nary([term], [var("ℓ"), roman(" = 1")], [roman("4")]),
    ]


def selection_frequency():
    indicator_set = subsup([var("T")], [var("b")], [var("j")])
    indicator = [roman("I"), delimiter([var("k"), roman(" ∈ "), indicator_set])]
    return [
        _indexed("π", "kj"), roman(" = "), fraction([roman("1")], [var("B")]),
        nary(indicator, [var("b"), roman(" = 1")], [var("B")]),
    ]


def oav_formula():
    return [
        subscript([roman("OAV")], [var("k")]), roman(" = "),
        fraction([_indexed("c", "k")], [_indexed("τ", "k")]),
        roman(",   "), _indexed("τ", "k"), roman(" > 0"),
    ]


FORMULAS = (
    model_relation,
    rank_correlation,
    graph_elastic_objective,
    conditional_importance,
    rrf_score,
    selection_frequency,
    oav_formula,
)
