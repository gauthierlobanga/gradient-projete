#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient projeté avancé pour problèmes quadratiques en dimension 2.

Résout:
    min f(x1, x2) = a*x1^2 + b*x2^2 + c*x1 + d*x2
    s.c. h_i(x) = alpha_i^T x + beta_i >= 0

Le programme privilégie la lisibilité mathématique: chaque itération affiche le
gradient, les contraintes actives, la projection, la recherche du pas exact et
les conditions KKT.
"""

from __future__ import annotations

import argparse
import math
import sys
import traceback
from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Optional

import numpy as np


EPSILON = 1e-8
MAX_ITER_DEFAULT = 50
FRACTION_DENOMINATOR = 1000


class Style:
    """Codes ANSI simples pour un affichage terminal moderne."""

    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    END = "\033[0m"


class ResolutionError(Exception):
    """Erreur métier pendant la résolution."""


class ValidationError(ResolutionError):
    """Données d'entrée invalides."""


class InfeasibleStartError(ResolutionError):
    """Le point initial ne satisfait pas les contraintes."""


class UnboundedProblemError(ResolutionError):
    """La fonction est non bornée dans une direction admissible."""


def finite_float(value: float, name: str) -> float:
    """Valide qu'une valeur est un réel fini."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} doit être un nombre réel.") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} doit être fini.")
    return result


def format_decimal(value: float, digits: int = 8) -> str:
    """Affiche proprement un flottant."""
    value = 0.0 if abs(value) < EPSILON else float(value)
    return f"{value:.{digits}g}"


def to_frac(value: float, max_denominator: int = FRACTION_DENOMINATOR) -> str:
    """Convertit un flottant en fraction courte quand c'est pertinent."""
    value = 0.0 if abs(value) < EPSILON else float(value)
    frac = Fraction(value).limit_denominator(max_denominator)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def format_scalar(value: float) -> str:
    """Affiche décimal + fraction si la fraction est informative."""
    dec = format_decimal(value)
    frac = to_frac(value)
    return dec if dec == frac else f"{dec} ({frac})"


def format_vector(vector: np.ndarray, name: str = "x") -> str:
    """Retourne une représentation compacte d'un vecteur colonne 2D."""
    return f"{name} = ({format_scalar(vector[0])}, {format_scalar(vector[1])})^T"


def format_matrix(matrix: np.ndarray, name: str) -> str:
    """Retourne une matrice 2D ou m x 2 en style lisible."""
    rows = []
    for row in matrix:
        rows.append("[ " + "  ".join(f"{format_decimal(v, 6):>10}" for v in row) + " ]")
    return f"{name} =\n" + "\n".join(rows)


def line(title: str = "", char: str = "=", width: int = 96) -> None:
    """Affiche une séparation avec titre optionnel."""
    if title:
        print(f"\n{Style.BOLD}{char * width}{Style.END}")
        print(f"{Style.BOLD}{title.center(width)}{Style.END}")
        print(f"{Style.BOLD}{char * width}{Style.END}")
    else:
        print(char * width)


def section(title: str) -> None:
    """Affiche un sous-titre."""
    print(f"\n{Style.BOLD}{Style.CYAN}{title}{Style.END}")


def print_table(headers: list[str], rows: Iterable[Iterable[object]]) -> None:
    """Affiche un tableau texte sans dépendance externe."""
    str_rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)
    for row in str_rows:
        print("| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |")
    print(sep)


def norm(vector: np.ndarray) -> float:
    """Norme euclidienne sous forme float Python."""
    return float(np.linalg.norm(vector))


@dataclass(frozen=True)
class ProblemeQuadratique:
    """Fonction quadratique diagonale en dimension 2."""

    a: float
    b: float
    c: float
    d: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", finite_float(self.a, "a"))
        object.__setattr__(self, "b", finite_float(self.b, "b"))
        object.__setattr__(self, "c", finite_float(self.c, "c"))
        object.__setattr__(self, "d", finite_float(self.d, "d"))
        if self.a < -EPSILON or self.b < -EPSILON:
            raise ValidationError(
                "La version convexe exige a >= 0 et b >= 0. "
                "Sinon le minimum global peut ne pas exister."
            )

    @property
    def q(self) -> np.ndarray:
        return np.diag([2.0 * self.a, 2.0 * self.b])

    @property
    def p(self) -> np.ndarray:
        return np.array([self.c, self.d], dtype=float)

    def valeur(self, x: np.ndarray) -> float:
        return float(self.a * x[0] ** 2 + self.b * x[1] ** 2 + self.c * x[0] + self.d * x[1])

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.q @ x + self.p

    def phi_coefficients(self, x: np.ndarray, direction: np.ndarray) -> tuple[float, float, float]:
        """Coefficients de phi(lambda)=f(x+lambda*d)=A lambda²+B lambda+C."""
        A = 0.5 * float(direction @ self.q @ direction)
        B = float(self.gradient(x) @ direction)
        C = self.valeur(x)
        return A, B, C

    def afficher(self) -> str:
        terms: list[str] = []
        for coeff, label in ((self.a, "x1^2"), (self.b, "x2^2")):
            if abs(coeff) >= EPSILON:
                terms.append(f"{to_frac(coeff)}*{label}")
        for coeff, label in ((self.c, "x1"), (self.d, "x2")):
            if abs(coeff) >= EPSILON:
                sign = "+" if coeff > 0 else "-"
                terms.append(f"{sign} {to_frac(abs(coeff))}*{label}")
        return " ".join(terms) if terms else "0"


@dataclass(frozen=True)
class Contrainte:
    """Contrainte linéaire h_i(x)=alpha_i^T x + beta >= 0."""

    idx: int
    alpha: np.ndarray
    beta: float

    def __post_init__(self) -> None:
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.shape != (2,):
            raise ValidationError(f"La contrainte h{self.idx} doit avoir deux coefficients alpha.")
        if not np.all(np.isfinite(alpha)):
            raise ValidationError(f"Les coefficients de h{self.idx} doivent être finis.")
        if norm(alpha) < EPSILON:
            raise ValidationError(f"La contrainte h{self.idx} a un vecteur alpha nul.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", finite_float(self.beta, f"beta de h{self.idx}"))

    def valeur(self, x: np.ndarray) -> float:
        return float(self.alpha @ x + self.beta)

    def est_active(self, x: np.ndarray, tol: float) -> bool:
        return abs(self.valeur(x)) <= tol

    def pas_limite(self, x: np.ndarray, direction: np.ndarray, tol: float) -> Optional[float]:
        """
        Calcule le pas positif où h_i devient nulle.

        Comme h_i(x+lambda*d)=h_i(x)+lambda*alpha_i^T d, la contrainte borne
        le déplacement seulement si alpha_i^T d < 0.
        """
        h0 = self.valeur(x)
        pente = float(self.alpha @ direction)
        if pente >= -tol:
            return None
        lam = h0 / (-pente)
        return lam if lam >= tol else 0.0

    def texte(self) -> str:
        return f"h{self.idx}(x) = {to_frac(self.alpha[0])}*x1 + {to_frac(self.alpha[1])}*x2 + {to_frac(self.beta)} >= 0"


@dataclass
class DirectionInfo:
    direction: Optional[np.ndarray]
    working_set: list[Contrainte]
    projection: np.ndarray
    projected_gradient: np.ndarray
    multipliers: dict[int, float]
    status: str
    explanation: str


@dataclass
class PasInfo:
    lambda_max: float
    lambda_star: float
    blocking_constraint: Optional[Contrainte]
    candidates: list[tuple[int, float]]
    phi: tuple[float, float, float]
    unbounded: bool = False


class GradientProjete:
    """Solveur actif/projeté robuste pour quadratiques convexes 2D."""

    def __init__(
        self,
        probleme: ProblemeQuadratique,
        contraintes: list[Contrainte],
        x0: np.ndarray,
        max_iter: int = MAX_ITER_DEFAULT,
        tol: float = EPSILON,
    ):
        self.probleme = probleme
        self.contraintes = contraintes
        self.x0 = self._validate_point(x0, "x0")
        self.max_iter = self._validate_iterations(max_iter)
        self.tol = finite_float(tol, "tol")
        if self.tol <= 0:
            raise ValidationError("La tolérance doit être strictement positive.")
        self.historique: list[dict[str, object]] = []
        self._validate_feasible_start(self.x0)

    def _validate_iterations(self, max_iter: int) -> int:
        try:
            value = int(max_iter)
        except (TypeError, ValueError) as exc:
            raise ValidationError("max_iter doit être un entier.") from exc
        if value < 1 or value > 10000:
            raise ValidationError("max_iter doit être compris entre 1 et 10000.")
        return value

    def _validate_point(self, x: np.ndarray, name: str) -> np.ndarray:
        point = np.asarray(x, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValidationError(f"{name} doit être un vecteur réel fini de dimension 2.")
        return point

    def _validate_feasible_start(self, x: np.ndarray) -> None:
        violated = [(c, c.valeur(x)) for c in self.contraintes if c.valeur(x) < -self.tol]
        if violated:
            details = ", ".join(f"h{c.idx}={format_decimal(value)}" for c, value in violated)
            raise InfeasibleStartError(f"Point initial non admissible: {details}.")

    def _active_inactive(self, x: np.ndarray) -> tuple[list[Contrainte], list[Contrainte]]:
        active: list[Contrainte] = []
        inactive: list[Contrainte] = []
        for constraint in self.contraintes:
            if constraint.est_active(x, self.tol):
                active.append(constraint)
            else:
                inactive.append(constraint)
        return active, inactive

    def _matrix_a(self, constraints: list[Contrainte]) -> np.ndarray:
        if not constraints:
            return np.empty((0, 2), dtype=float)
        return np.vstack([constraint.alpha for constraint in constraints])

    def _projection(self, constraints: list[Contrainte]) -> np.ndarray:
        """Projection orthogonale sur le noyau de A."""
        if not constraints:
            return np.eye(2)
        A = self._matrix_a(constraints)
        return np.eye(2) - A.T @ np.linalg.pinv(A @ A.T) @ A

    def _multipliers(self, gradient: np.ndarray, constraints: list[Contrainte]) -> dict[int, float]:
        """Résout A^T mu = gradient au sens des moindres carrés."""
        if not constraints:
            return {}
        A = self._matrix_a(constraints)
        mu, *_ = np.linalg.lstsq(A.T, gradient, rcond=None)
        return {constraint.idx: float(mu[i]) for i, constraint in enumerate(constraints)}

    def _direction(self, x: np.ndarray, active: list[Contrainte]) -> DirectionInfo:
        gradient = self.probleme.gradient(x)
        projection = self._projection(active)
        projected_gradient = projection @ gradient

        if norm(projected_gradient) > self.tol:
            direction = -projected_gradient
            return DirectionInfo(
                direction=direction,
                working_set=active,
                projection=projection,
                projected_gradient=projected_gradient,
                multipliers={},
                status="DESCENTE_PROJETEE",
                explanation="Le gradient projeté est non nul: Z = -P*grad f.",
            )

        multipliers = self._multipliers(gradient, active)
        if not active:
            if norm(gradient) <= self.tol:
                return DirectionInfo(
                    direction=None,
                    working_set=[],
                    projection=projection,
                    projected_gradient=projected_gradient,
                    multipliers={},
                    status="OPTIMAL",
                    explanation="Gradient nul sans contrainte active.",
                )
            return DirectionInfo(
                direction=-gradient,
                working_set=[],
                projection=projection,
                projected_gradient=projected_gradient,
                multipliers={},
                status="DESCENTE_LIBRE",
                explanation="Aucune contrainte active: Z = -grad f.",
            )

        negative = [(idx, mu) for idx, mu in multipliers.items() if mu < -self.tol]
        if not negative:
            return DirectionInfo(
                direction=None,
                working_set=active,
                projection=projection,
                projected_gradient=projected_gradient,
                multipliers=multipliers,
                status="OPTIMAL",
                explanation="KKT vérifié: P*grad f = 0 et tous les multiplicateurs mu_i >= 0.",
            )

        released_idx, released_mu = min(negative, key=lambda item: item[1])
        reduced_active = [constraint for constraint in active if constraint.idx != released_idx]
        reduced_projection = self._projection(reduced_active)
        reduced_projected_gradient = reduced_projection @ gradient
        direction = -reduced_projected_gradient
        if norm(direction) <= self.tol:
            direction = -gradient
        return DirectionInfo(
            direction=direction,
            working_set=reduced_active,
            projection=reduced_projection,
            projected_gradient=reduced_projected_gradient,
            multipliers=multipliers,
            status="CONTRAINTE_RELACHEE",
            explanation=(
                f"mu_{released_idx} = {format_decimal(released_mu)} < 0; "
                f"la contrainte h{released_idx} est relâchée pour reprendre la descente."
            ),
        )

    def _pas(self, x: np.ndarray, direction: np.ndarray, inactive: list[Contrainte]) -> PasInfo:
        candidates: list[tuple[int, float]] = []
        lambda_max = math.inf
        blocking: Optional[Contrainte] = None
        for constraint in inactive:
            lam = constraint.pas_limite(x, direction, self.tol)
            if lam is None:
                continue
            candidates.append((constraint.idx, lam))
            if lam < lambda_max:
                lambda_max = lam
                blocking = constraint

        A, B, C = self.probleme.phi_coefficients(x, direction)
        if A > self.tol:
            lambda_unconstrained = max(0.0, -B / (2.0 * A))
            lambda_star = lambda_unconstrained if math.isinf(lambda_max) else min(lambda_unconstrained, lambda_max)
        elif B < -self.tol:
            if math.isinf(lambda_max):
                return PasInfo(lambda_max, math.inf, blocking, candidates, (A, B, C), unbounded=True)
            lambda_star = lambda_max
        else:
            lambda_star = 0.0

        return PasInfo(lambda_max, float(lambda_star), blocking, candidates, (A, B, C))

    def _print_problem(self) -> None:
        line("RESOLUTION PAR GRADIENT PROJETE AVANCE", "=", 96)
        print(f"{Style.BOLD}Objectif:{Style.END} min f(x) = {self.probleme.afficher()}")
        print(f"{Style.BOLD}Forme matricielle:{Style.END} f(x)=1/2*x^T*Q*x+p^T*x")
        print(format_matrix(self.probleme.q, "Q"))
        print(f"p = ({format_scalar(self.probleme.p[0])}, {format_scalar(self.probleme.p[1])})^T")
        section("Contraintes")
        if not self.contraintes:
            print("Aucune contrainte.")
        else:
            print_table(
                ["i", "h_i(x) >= 0"],
                [[constraint.idx, constraint.texte()] for constraint in self.contraintes],
            )
        print(f"{Style.BOLD}Point initial:{Style.END} {format_vector(self.x0, 'X0')}")
        print(f"{Style.BOLD}Paramètres:{Style.END} tol = {self.tol:g}, max_iter = {self.max_iter}")

    def _print_iteration(
        self,
        k: int,
        x: np.ndarray,
        active: list[Contrainte],
        inactive: list[Contrainte],
        direction_info: DirectionInfo,
        pas_info: Optional[PasInfo],
        x_new: Optional[np.ndarray],
    ) -> None:
        gradient = self.probleme.gradient(x)
        line(f"ITERATION {k}", "-", 96)
        print(f"{Style.BOLD}Point courant:{Style.END} {format_vector(x, 'Xk')}")
        print(f"{Style.BOLD}Valeur:{Style.END} f(Xk) = {format_decimal(self.probleme.valeur(x), 10)}")
        print(f"{Style.BOLD}Gradient:{Style.END} {format_vector(gradient, 'grad f(Xk)')}")

        rows = []
        for constraint in self.contraintes:
            value = constraint.valeur(x)
            state = "active" if constraint in active else "inactive"
            rows.append([f"h{constraint.idx}", format_decimal(value, 10), state])
        if rows:
            print_table(["Contrainte", "h_i(Xk)", "Etat"], rows)

        print(f"{Style.BOLD}Ensemble actif:{Style.END} {[c.idx for c in active] or 'aucun'}")
        print(f"{Style.BOLD}Ensemble de travail:{Style.END} {[c.idx for c in direction_info.working_set] or 'aucun'}")
        print(format_matrix(direction_info.projection, "P"))
        print(f"{Style.BOLD}Gradient projete:{Style.END} {format_vector(direction_info.projected_gradient, 'P*grad f')}")
        print(f"{Style.BOLD}Diagnostic:{Style.END} {direction_info.explanation}")

        if direction_info.multipliers:
            print_table(
                ["Multiplicateur", "Valeur", "KKT"],
                [
                    [f"mu_{idx}", format_decimal(mu, 10), "OK" if mu >= -self.tol else "negatif"]
                    for idx, mu in direction_info.multipliers.items()
                ],
            )

        if direction_info.direction is None:
            print(f"{Style.GREEN}{Style.BOLD}Arret:{Style.END} conditions d'optimalite satisfaites.")
            return

        print(f"{Style.BOLD}Direction:{Style.END} {format_vector(direction_info.direction, 'Z')}")
        print(f"{Style.BOLD}Test descente:{Style.END} grad f(Xk)^T*Z = {format_decimal(float(gradient @ direction_info.direction), 10)}")

        if pas_info is None:
            return
        A, B, C = pas_info.phi
        print(f"{Style.BOLD}Recherche lineaire exacte:{Style.END} phi(lambda)=A*lambda^2+B*lambda+C")
        print(f"A = {format_decimal(A, 10)}, B = {format_decimal(B, 10)}, C = {format_decimal(C, 10)}")
        if pas_info.candidates:
            print_table(
                ["Blocage possible", "lambda"],
                [[f"h{idx}", format_scalar(lam)] for idx, lam in pas_info.candidates],
            )
        else:
            print("Aucune contrainte inactive ne borne la direction.")

        if pas_info.unbounded:
            print(f"{Style.FAIL}{Style.BOLD}Probleme non borne:{Style.END} phi decroit sans limite dans une direction admissible.")
            return

        if math.isinf(pas_info.lambda_max):
            print("lambda_max = +inf")
        else:
            blocker = f"h{pas_info.blocking_constraint.idx}" if pas_info.blocking_constraint else "aucune"
            print(f"lambda_max = {format_scalar(pas_info.lambda_max)} ({blocker})")
        print(f"{Style.BOLD}Pas retenu:{Style.END} lambda* = {format_scalar(pas_info.lambda_star)}")
        if x_new is not None:
            delta = self.probleme.valeur(x_new) - self.probleme.valeur(x)
            print(f"{Style.BOLD}Nouveau point:{Style.END} {format_vector(x_new, 'Xk+1')}")
            print(f"{Style.BOLD}Variation:{Style.END} f(Xk+1)-f(Xk) = {format_decimal(delta, 10)}")

    def _print_final(self, x: np.ndarray, status: str) -> None:
        active, _ = self._active_inactive(x)
        gradient = self.probleme.gradient(x)
        multipliers = self._multipliers(gradient, active)
        line("RESULTAT FINAL", "=", 96)
        print(f"{Style.BOLD}{Style.GREEN}Statut:{Style.END} {status}")
        print(f"{Style.BOLD}Point obtenu:{Style.END} {format_vector(x, 'X*')}")
        print(f"{Style.BOLD}Valeur optimale:{Style.END} f(X*) = {format_decimal(self.probleme.valeur(x), 12)}")
        print(f"{Style.BOLD}Gradient final:{Style.END} {format_vector(gradient, 'grad f(X*)')}")
        print(f"{Style.BOLD}Contraintes actives:{Style.END} {[c.idx for c in active] or 'aucune'}")
        if self.contraintes:
            print_table(
                ["Contrainte", "h_i(X*)", "Etat"],
                [
                    [
                        f"h{constraint.idx}",
                        format_decimal(constraint.valeur(x), 12),
                        "active" if constraint in active else "inactive",
                    ]
                    for constraint in self.contraintes
                ],
            )
        if multipliers:
            print_table(
                ["Multiplicateur KKT", "Valeur"],
                [[f"mu_{idx}", format_decimal(mu, 12)] for idx, mu in multipliers.items()],
            )

    def resoudre(self, verbose: bool = True) -> tuple[np.ndarray, float, list[dict[str, object]]]:
        """Exécute l'algorithme et retourne (x*, f(x*), historique)."""
        x = self.x0.copy()
        self.historique = []
        status = "ITERATIONS_MAX_ATTEINTES"

        if verbose:
            self._print_problem()

        for k in range(self.max_iter):
            active, inactive = self._active_inactive(x)
            direction_info = self._direction(x, active)

            if direction_info.direction is None:
                self.historique.append(
                    {
                        "iteration": k,
                        "x": x.copy(),
                        "f": self.probleme.valeur(x),
                        "gradient": self.probleme.gradient(x),
                        "active": [c.idx for c in active],
                        "direction": None,
                        "lambda": None,
                        "status": direction_info.status,
                    }
                )
                if verbose:
                    self._print_iteration(k, x, active, inactive, direction_info, None, None)
                status = "OPTIMALITE_KKT"
                break

            if norm(direction_info.direction) <= self.tol:
                status = "DIRECTION_NULLE"
                if verbose:
                    self._print_iteration(k, x, active, inactive, direction_info, None, None)
                break

            pas_info = self._pas(x, direction_info.direction, inactive)
            if pas_info.unbounded:
                if verbose:
                    self._print_iteration(k, x, active, inactive, direction_info, pas_info, None)
                raise UnboundedProblemError("La fonction objectif est non bornee sur l'ensemble admissible.")

            if pas_info.lambda_star <= self.tol:
                status = "PAS_TROP_PETIT"
                if verbose:
                    self._print_iteration(k, x, active, inactive, direction_info, pas_info, x)
                break

            x_new = x + pas_info.lambda_star * direction_info.direction
            x_new[np.abs(x_new) < self.tol] = 0.0
            self._validate_feasible_start(x_new)

            self.historique.append(
                {
                    "iteration": k,
                    "x": x.copy(),
                    "f": self.probleme.valeur(x),
                    "gradient": self.probleme.gradient(x),
                    "active": [c.idx for c in active],
                    "working_set": [c.idx for c in direction_info.working_set],
                    "direction": direction_info.direction.copy(),
                    "lambda": pas_info.lambda_star,
                    "lambda_max": pas_info.lambda_max,
                    "blocking_constraint": pas_info.blocking_constraint.idx if pas_info.blocking_constraint else None,
                    "status": direction_info.status,
                }
            )

            if verbose:
                self._print_iteration(k, x, active, inactive, direction_info, pas_info, x_new)

            objective_delta = abs(self.probleme.valeur(x_new) - self.probleme.valeur(x))
            step_norm = norm(x_new - x)
            x = x_new
            if step_norm <= self.tol or objective_delta <= self.tol:
                status = "CONVERGENCE_NUMERIQUE"
                break

        if verbose:
            self._print_final(x, status)
        return x, self.probleme.valeur(x), self.historique


def read_float(prompt: str) -> float:
    """Saisie robuste d'un réel fini."""
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            return finite_float(raw, prompt)
        except ValidationError as exc:
            print(f"{Style.FAIL}Erreur: {exc}{Style.END}")


def read_int(prompt: str, min_value: int, max_value: int, default: Optional[int] = None) -> int:
    """Saisie robuste d'un entier avec valeur par défaut optionnelle."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"{Style.FAIL}Erreur: veuillez entrer un entier.{Style.END}")
            continue
        if min_value <= value <= max_value:
            return value
        print(f"{Style.FAIL}Erreur: valeur attendue entre {min_value} et {max_value}.{Style.END}")


def read_constraints() -> list[Contrainte]:
    """Saisie interactive des contraintes."""
    section("Saisie des contraintes")
    print("Format: h_i(x)=alpha1*x1 + alpha2*x2 + beta >= 0")
    count = read_int("Nombre de contraintes", 0, 50, default=0)
    constraints: list[Contrainte] = []
    for idx in range(1, count + 1):
        while True:
            print(f"\n{Style.BOLD}Contrainte h{idx}{Style.END}")
            alpha1 = read_float("  alpha1: ")
            alpha2 = read_float("  alpha2: ")
            beta = read_float("  beta: ")
            try:
                constraints.append(Contrainte(idx, np.array([alpha1, alpha2]), beta))
                break
            except ValidationError as exc:
                print(f"{Style.FAIL}Erreur: {exc}{Style.END}")
    return constraints


def read_start_point(constraints: list[Contrainte], tol: float) -> np.ndarray:
    """Saisie du point de départ avec contrôle d'admissibilité."""
    section("Point de depart")
    while True:
        x1 = read_float("x1 initial: ")
        x2 = read_float("x2 initial: ")
        x0 = np.array([x1, x2], dtype=float)
        violated = [(c, c.valeur(x0)) for c in constraints if c.valeur(x0) < -tol]
        if not violated:
            return x0
        print(f"{Style.FAIL}Point non admissible.{Style.END}")
        print_table(
            ["Contrainte violee", "Valeur"],
            [[f"h{c.idx}", format_decimal(value, 10)] for c, value in violated],
        )


def build_demo() -> tuple[ProblemeQuadratique, list[Contrainte], np.ndarray]:
    """Exemple stable pour tester rapidement le solveur."""
    problem = ProblemeQuadratique(a=1, b=1, c=-4, d=-6)
    constraints = [
        Contrainte(1, np.array([1, 0], dtype=float), 0),      # x1 >= 0
        Contrainte(2, np.array([0, 1], dtype=float), 0),      # x2 >= 0
        Contrainte(3, np.array([-1, -1], dtype=float), 5),    # x1+x2 <= 5
    ]
    x0 = np.array([0, 0], dtype=float)
    return problem, constraints, x0


def interactive_problem() -> tuple[ProblemeQuadratique, list[Contrainte], np.ndarray, int, float]:
    """Construit un problème depuis le terminal."""
    print(f"{Style.BOLD}\nALGORITHME DU GRADIENT PROJETE{Style.END}")
    print("Resolution de min f(x1,x2)=a*x1^2+b*x2^2+c*x1+d*x2 sous h_i(x)>=0.\n")

    while True:
        section("Fonction objectif")
        a = read_float("Coefficient a de x1^2: ")
        b = read_float("Coefficient b de x2^2: ")
        c = read_float("Coefficient c de x1: ")
        d = read_float("Coefficient d de x2: ")
        try:
            problem = ProblemeQuadratique(a, b, c, d)
            break
        except ValidationError as exc:
            print(f"{Style.FAIL}Erreur: {exc}{Style.END}")

    constraints = read_constraints()
    tol = EPSILON
    x0 = read_start_point(constraints, tol)
    section("Parametres")
    max_iter = read_int("Nombre maximal d'iterations", 1, 10000, default=MAX_ITER_DEFAULT)
    return problem, constraints, x0, max_iter, tol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solveur avance du gradient projete pour quadratiques convexes 2D."
    )
    parser.add_argument("--demo", action="store_true", help="Lance un exemple complet sans saisie interactive.")
    parser.add_argument("--quiet", action="store_true", help="Masque les details et affiche seulement le resultat.")
    parser.add_argument("--max-iter", type=int, default=MAX_ITER_DEFAULT, help="Nombre maximal d'iterations.")
    parser.add_argument("--tol", type=float, default=EPSILON, help="Tolerance numerique.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.demo:
            problem, constraints, x0 = build_demo()
            max_iter = args.max_iter
            tol = args.tol
        else:
            problem, constraints, x0, max_iter, tol = interactive_problem()

        solver = GradientProjete(problem, constraints, x0, max_iter=max_iter, tol=tol)
        x_opt, f_opt, _ = solver.resoudre(verbose=not args.quiet)

        if args.quiet:
            print(format_vector(x_opt, "X*"))
            print(f"f(X*) = {format_decimal(f_opt, 12)}")
        print(f"\n{Style.GREEN}{Style.BOLD}Resolution terminee avec succes.{Style.END}")
        return 0
    except ResolutionError as exc:
        print(f"\n{Style.FAIL}{Style.BOLD}Erreur de resolution:{Style.END} {exc}")
        return 2
    except KeyboardInterrupt:
        print(f"\n{Style.WARNING}Operation interrompue par l'utilisateur.{Style.END}")
        return 130
    except Exception:
        print(f"\n{Style.FAIL}{Style.BOLD}Erreur inattendue:{Style.END}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
