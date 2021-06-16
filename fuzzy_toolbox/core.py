import matplotlib.pyplot as plt
import numpy as np


def format_plot(title=None, view_xaxis=True, view_yaxis=False):

    plt.gca().set_ylim(-0.05, 1.05)

    plt.gca().spines["bottom"].set_visible(True)
    plt.gca().spines["left"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    plt.gca().spines["bottom"].set_color("gray")

    plt.gca().get_yaxis().set_visible(False)
    if view_yaxis == "left":
        plt.gca().get_yaxis().set_visible(True)
    if view_yaxis == "right":
        plt.gca().get_yaxis().set_visible(True)
        plt.gca().yaxis.tick_right()

    plt.gca().get_xaxis().set_visible(view_xaxis)

    if title is not None:
        plt.gca().set_title(title)


def plot_fuzzyvariable(
    universe, memberships, labels, title, fmt, linewidth, view_xaxis, view_yaxis
):
    #
    for label, membership in zip(labels, memberships):
        plt.gca().plot(universe, membership, fmt, label=label, linewidth=linewidth)
    plt.gca().legend()
    #
    format_plot(
        title=title,
        view_xaxis=view_xaxis,
        view_yaxis=view_yaxis,
    )

    # plt.gca().spines["left"].set_color("lightgray")


def plot_crisp_input(
    value, universe, membership, name, view_xaxis=True, view_yaxis="left"
):

    plt.gca().plot(universe, membership, "-k", linewidth=1)

    membership_value = np.interp(
        x=value,
        xp=universe,
        fp=membership,
    )
    membership = np.where(membership <= membership_value, membership, membership_value)
    plt.gca().fill_between(universe, membership, color="gray", alpha=0.7)

    if name is None:
        title = None
    else:
        title = "{} = {}".format(name, value)

    format_plot(
        title=title,
        view_xaxis=view_xaxis,
        view_yaxis=view_yaxis,
    )

    plt.gca().vlines(x=value, ymin=-0.0, ymax=1.0, color="red", linewidth=2)


def plot_fuzzy_input(
    value, universe, membership, name, view_xaxis=True, view_yaxis="left"
):

    plt.gca().plot(universe, membership, "-k", linewidth=1)

    plt.gca().fill_between(universe, value, color="gray", alpha=0.7)

    format_plot(
        title=name,
        view_xaxis=view_xaxis,
        view_yaxis=view_yaxis,
    )


def apply_modifiers(membership, modifiers):
    def slightly(u):
        plus_u = np.power(u, 1.25)
        not_very_u = 1 - np.power(u, 2)
        u = np.where(u < not_very_u, plus_u, not_very_u)
        u = u / np.max(u)
        u = np.where(u <= 0.5, u ** 2, 1 - 2 * (1 - u) ** 2)
        return u

    fn = {
        "VERY": lambda u: np.power(u, 2),
        "SOMEWHAT": lambda u: np.power(u, 1.0 / 3.0),
        "MORE_OR_LESS": lambda u: np.power(u, 0.5),
        "EXTREMELY": lambda u: np.power(u, 3),
        "PLUS": lambda u: np.power(u, 1.25),
        "INTENSIFY": lambda u: np.where(
            u <= 0.5, np.power(u, 2), 1 - 2 * np.power(1 - u, 2)
        ),
        "NORM": lambda u: u / np.max(u),
        "NOT": lambda u: 1 - u,
        "SLIGHTLY": lambda u: slightly(u),
    }

    membership = membership.copy()
    modifiers = modifiers.copy()
    modifiers.reverse()

    for modifier in modifiers:
        membership = fn[modifier](membership)

    return membership
