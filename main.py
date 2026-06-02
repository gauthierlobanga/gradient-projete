#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradient projeté – Version interactive corrigée
Résout min f(x1,x2)=a*x1^2+b*x2^2+c*x1+d*x2
s.c. h_i(x)=alpha_i^T x + beta_i >= 0
"""

from __future__ import annotations

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
    pass


class ValidationError(ResolutionError):
    pass


class InfeasibleStartError(ResolutionError):
    pass


class UnboundedProblemError(ResolutionError):
    pass


def finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{name} doit être un nombre réel.") from exc
    if not math.isfinite(result):
        raise ValidationError(f"{name} doit être fini.")
    return result


def format_decimal(value, digits=8):
    value = 0.0 if abs(value) < EPSILON else float(value)
    return f"{value:.{digits}g}"


def to_frac(value, max_denominator=FRACTION_DENOMINATOR):
    value = 0.0 if abs(value) < EPSILON else float(value)
    frac = Fraction(value).limit_denominator(max_denominator)
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator}"


def format_scalar(value):
    dec = format_decimal(value)
    frac = to_frac(value)
    return dec if dec == frac else f"{dec} ({frac})"


def format_vector(vector, name="x"):
    return f"{name} = ({format_scalar(vector[0])}, {format_scalar(vector[1])})^T"


def format_matrix(matrix, name):
    rows = []
    for row in matrix:
        rows.append("[ " + "  ".join(f"{format_decimal(v, 6):>10}" for v in row) + " ]")
    return f"{name} =\n" + "\n".join(rows)


def line(title="", char="=", width=96):
    if title:
        print(f"\n{Style.BOLD}{char * width}{Style.END}")
        print(f"{Style.BOLD}{title.center(width)}{Style.END}")
        print(f"{Style.BOLD}{char * width}{Style.END}")
    else:
        print(char * width)


def section(title):
    print(f"\n{Style.BOLD}{Style.CYAN}{title}{Style.END}")


def print_table(headers, rows):
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


def norm(vector):
    return float(np.linalg.norm(vector))


@dataclass(frozen=True)
class ProblemeQuadratique:
    a: float
    b: float
    c: float
    d: float

    def __post_init__(self):
        object.__setattr__(self, "a", finite_float(self.a, "a"))
        object.__setattr__(self, "b", finite_float(self.b, "b"))
        object.__setattr__(self, "c", finite_float(self.c, "c"))
        object.__setattr__(self, "d", finite_float(self.d, "d"))
        if self.a < -EPSILON or self.b < -EPSILON:
            print(f"{Style.WARNING}Attention : fonction non convexe (a={self.a}, b={self.b}){Style.END}")

    def valeur(self, x):
        return self.a * x[0] ** 2 + self.b * x[1] ** 2 + self.c * x[0] + self.d * x[1]

    def gradient(self, x):
        return np.array([2 * self.a * x[0] + self.c, 2 * self.b * x[1] + self.d])

    def phi_coefficients(self, x, direction):
        # f(x+λd) = a(x1+λd1)^2 + b(x2+λd2)^2 + c(x1+λd1) + d(x2+λd2)
        A = self.a * direction[0] ** 2 + self.b * direction[1] ** 2
        B = 2 * self.a * x[0] * direction[0] + 2 * self.b * x[1] * direction[1] + self.c * direction[0] + self.d * direction[1]
        C = self.valeur(x)
        return A, B, C

    def afficher(self):
        terms = []
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
    idx: int
    alpha: np.ndarray
    beta: float

    def __post_init__(self):
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.shape != (2,):
            raise ValidationError(f"Contrainte h{self.idx} : alpha doit être de dimension 2.")
        if norm(alpha) < EPSILON:
            raise ValidationError(f"Contrainte h{self.idx} : alpha nul.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", finite_float(self.beta, f"beta de h{self.idx}"))

    def valeur(self, x):
        return float(self.alpha @ x + self.beta)

    def est_active(self, x, tol):
        return abs(self.valeur(x)) <= tol

    def pas_limite(self, x, direction, tol):
        """Retourne le pas λ > 0 où h devient nulle, ou None si pas de blocage."""
        h0 = self.valeur(x)
        pente = float(self.alpha @ direction)
        if pente >= -tol:   # direction sortante ou tangentielle
            return None
        lam = -h0 / pente   # car h0 + λ pente = 0 => λ = -h0/pente
        return lam if lam > tol else 0.0

    def texte(self):
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
    def __init__(self, probleme, contraintes, x0, max_iter=MAX_ITER_DEFAULT, tol=EPSILON):
        self.probleme = probleme
        self.contraintes = contraintes
        self.x0 = self._validate_point(x0, "x0")
        self.max_iter = max(1, int(max_iter))
        self.tol = finite_float(tol, "tol")
        self.historique = []
        self._validate_feasible_start(self.x0)

    def _validate_point(self, x, name):
        point = np.asarray(x, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValidationError(f"{name} doit être un vecteur 2D réel.")
        return point

    def _validate_feasible_start(self, x):
        violated = [(c, c.valeur(x)) for c in self.contraintes if c.valeur(x) < -self.tol]
        if violated:
            print(f"{Style.FAIL}Point initial non admissible :{Style.END}")
            for c, val in violated:
                print(f"  {c.texte()} → valeur = {format_decimal(val)}")
            raise InfeasibleStartError("Choisissez un point satisfaisant toutes les contraintes.")

    def _active_inactive(self, x):
        active, inactive = [], []
        for c in self.contraintes:
            if c.est_active(x, self.tol):
                active.append(c)
            else:
                inactive.append(c)
        return active, inactive

    def _projection(self, constraints):
        if not constraints:
            return np.eye(2)
        A = np.vstack([c.alpha for c in constraints])
        return np.eye(2) - A.T @ np.linalg.pinv(A @ A.T) @ A

    def _multipliers(self, gradient, constraints):
        if not constraints:
            return {}
        A = np.vstack([c.alpha for c in constraints])
        mu, *_ = np.linalg.lstsq(A.T, gradient, rcond=None)
        return {c.idx: float(mu[i]) for i, c in enumerate(constraints)}

    def _direction(self, x, active):
        g = self.probleme.gradient(x)
        P = self._projection(active)
        pg = P @ g

        if norm(pg) > self.tol:
            return DirectionInfo(
                direction=-pg,
                working_set=active,
                projection=P,
                projected_gradient=pg,
                multipliers={},
                status="DESCENTE_PROJETEE",
                explanation="Gradient projeté non nul → Z = -P·∇f"
            )

        # Projection nulle : point stationnaire dans le sous-espace actif
        if not active:
            if norm(g) <= self.tol:
                return DirectionInfo(None, [], P, pg, {}, "OPTIMAL", "Gradient nul, pas de contraintes")
            return DirectionInfo(-g, [], P, pg, {}, "DESCENTE_LIBRE", "Aucune contrainte active → Z = -∇f")

        # Calcul des multiplicateurs
        mu_dict = self._multipliers(g, active)
        negatives = [(idx, mu) for idx, mu in mu_dict.items() if mu < -self.tol]
        if not negatives:
            return DirectionInfo(None, active, P, pg, mu_dict, "OPTIMAL", "KKT vérifié (mu_i >= 0)")
        # Relâcher la plus négative
        idx_rel, _ = min(negatives, key=lambda x: x[1])
        new_active = [c for c in active if c.idx != idx_rel]
        P_new = self._projection(new_active)
        pg_new = P_new @ g
        if norm(pg_new) > self.tol:
            return DirectionInfo(-pg_new, new_active, P_new, pg_new, mu_dict, "CONTRAINTE_RELACHEE",
                                 f"mu_{idx_rel} < 0 → contrainte relâchée")
        else:
            return DirectionInfo(-g, new_active, P_new, pg_new, mu_dict, "DESCENTE_FORCEE", "Descente standard")

    def _pas(self, x, direction, inactive):
        candidates = []
        lambda_max = math.inf
        blocking = None
        for c in inactive:
            lam = c.pas_limite(x, direction, self.tol)
            if lam is not None:
                candidates.append((c.idx, lam))
                if lam < lambda_max:
                    lambda_max = lam
                    blocking = c

        A, B, C = self.probleme.phi_coefficients(x, direction)
        if A > self.tol:
            lambda_opt = max(0.0, -B / (2.0 * A))
            lambda_star = lambda_opt if math.isinf(lambda_max) else min(lambda_opt, lambda_max)
        elif B < -self.tol:
            if math.isinf(lambda_max):
                return PasInfo(lambda_max, math.inf, blocking, candidates, (A, B, C), unbounded=True)
            lambda_star = lambda_max
        else:
            lambda_star = 0.0
        return PasInfo(lambda_max, lambda_star, blocking, candidates, (A, B, C))

    def _print_problem(self):
        line("RÉSOLUTION PAR GRADIENT PROJETÉ", "=", 96)
        print(f"{Style.BOLD}Objectif :{Style.END} min f(x) = {self.probleme.afficher()}")
        section("Contraintes")
        if not self.contraintes:
            print("Aucune contrainte.")
        else:
            print_table(["i", "h_i(x) ≥ 0"], [[c.idx, c.texte()] for c in self.contraintes])
        print(f"{Style.BOLD}Point initial :{Style.END} {format_vector(self.x0, 'X0')}")
        print(f"{Style.BOLD}Paramètres :{Style.END} tol = {self.tol:g}, max_iter = {self.max_iter}")

    def _print_iteration(self, k, x, active, inactive, dinfo, pinfo, x_new):
        g = self.probleme.gradient(x)
        line(f"ITÉRATION {k}", "-", 96)
        print(f"{Style.BOLD}Point courant :{Style.END} {format_vector(x, 'Xk')}")
        print(f"{Style.BOLD}f(Xk) ={Style.END} {format_decimal(self.probleme.valeur(x), 10)}")
        print(f"{Style.BOLD}∇f(Xk) ={Style.END} {format_vector(g, '')}")
        rows = [[f"h{c.idx}", format_decimal(c.valeur(x), 10), "active" if c in active else "inactive"] for c in self.contraintes]
        print_table(["Contrainte", "h_i(Xk)", "État"], rows)
        print(f"{Style.BOLD}Actives :{Style.END} {[c.idx for c in active] or 'aucune'}")
        print(f"{Style.BOLD}Ensemble de travail :{Style.END} {[c.idx for c in dinfo.working_set] or 'aucune'}")
        print(format_matrix(dinfo.projection, "P"))
        print(f"{Style.BOLD}Gradient projeté :{Style.END} {format_vector(dinfo.projected_gradient, 'P·∇f')}")
        print(f"{Style.BOLD}Diagnostic :{Style.END} {dinfo.explanation}")
        if dinfo.multipliers:
            print_table(["μ", "Valeur", "KKT"],
                        [[f"μ_{idx}", format_decimal(mu, 10), "OK" if mu >= -self.tol else "négatif"] for idx, mu in dinfo.multipliers.items()])
        if dinfo.direction is None:
            print(f"{Style.GREEN}{Style.BOLD}Arrêt : conditions d'optimalité satisfaites.{Style.END}")
            return
        print(f"{Style.BOLD}Direction Z :{Style.END} {format_vector(dinfo.direction, 'Z')}")
        print(f"{Style.BOLD}Descente : ∇f·Z ={Style.END} {format_decimal(float(g @ dinfo.direction), 10)}")
        if pinfo is None:
            return
        A, B, C = pinfo.phi
        print(f"{Style.BOLD}φ(λ) = {format_decimal(A)} λ² + {format_decimal(B)} λ + {format_decimal(C)}{Style.END}")
        if pinfo.candidates:
            print_table(["Blocage", "λ"], [[f"h{idx}", format_scalar(lam)] for idx, lam in pinfo.candidates])
        else:
            print("Aucune contrainte inactive ne bloque.")
        if pinfo.unbounded:
            print(f"{Style.FAIL}Problème non borné dans cette direction.{Style.END}")
            return
        if math.isinf(pinfo.lambda_max):
            print("λ_max = +∞")
        else:
            print(f"λ_max = {format_scalar(pinfo.lambda_max)} (bloqué par h{pinfo.blocking_constraint.idx})")
        print(f"{Style.BOLD}Pas retenu λ* = {format_scalar(pinfo.lambda_star)}{Style.END}")
        if x_new is not None:
            delta = self.probleme.valeur(x_new) - self.probleme.valeur(x)
            print(f"{Style.BOLD}Nouveau point :{Style.END} {format_vector(x_new, 'Xk+1')}")
            print(f"Δf = {format_decimal(delta, 10)}")

    def _print_final(self, x, status):
        active, _ = self._active_inactive(x)
        g = self.probleme.gradient(x)
        mu = self._multipliers(g, active)
        line("RÉSULTAT FINAL", "=", 96)
        print(f"{Style.BOLD}{Style.GREEN}Statut :{Style.END} {status}")
        print(f"{Style.BOLD}Solution :{Style.END} {format_vector(x, 'X*')}")
        print(f"{Style.BOLD}f(X*) ={Style.END} {format_decimal(self.probleme.valeur(x), 12)}")
        print(f"{Style.BOLD}∇f(X*) ={Style.END} {format_vector(g, '')}")
        print(f"{Style.BOLD}Contraintes actives :{Style.END} {[c.idx for c in active] or 'aucune'}")
        print_table(["Contrainte", "h_i(X*)", "État"],
                    [[f"h{c.idx}", format_decimal(c.valeur(x), 12), "active" if c in active else "inactive"] for c in self.contraintes])
        if mu:
            print_table(["Multiplicateur KKT", "Valeur"], [[f"μ_{idx}", format_decimal(mu_val, 12)] for idx, mu_val in mu.items()])

    def resoudre(self, verbose=True):
        x = self.x0.copy()
        self.historique = []
        status = "MAX_ITER"

        if verbose:
            self._print_problem()

        for k in range(self.max_iter):
            active, inactive = self._active_inactive(x)
            dinfo = self._direction(x, active)

            if dinfo.direction is None:
                if verbose:
                    self._print_iteration(k, x, active, inactive, dinfo, None, None)
                status = "OPTIMAL"
                break

            if norm(dinfo.direction) <= self.tol:
                status = "DIR_NULLE"
                break

            pinfo = self._pas(x, dinfo.direction, inactive)
            if pinfo.unbounded:
                raise UnboundedProblemError("Fonction non bornée.")
            if pinfo.lambda_star <= self.tol:
                status = "PAS_TROP_PETIT"
                if verbose:
                    self._print_iteration(k, x, active, inactive, dinfo, pinfo, x)
                break

            x_new = x + pinfo.lambda_star * dinfo.direction
            x_new[np.abs(x_new) < self.tol] = 0.0

            self.historique.append({
                "k": k, "x": x.copy(), "f": self.probleme.valeur(x),
                "active": [c.idx for c in active], "Z": dinfo.direction.copy(), "λ": pinfo.lambda_star
            })

            if verbose:
                self._print_iteration(k, x, active, inactive, dinfo, pinfo, x_new)

            if norm(x_new - x) <= self.tol or abs(self.probleme.valeur(x_new) - self.probleme.valeur(x)) <= self.tol:
                status = "CONVERGENCE"
                x = x_new
                break
            x = x_new

        if verbose:
            self._print_final(x, status)
        return x, self.probleme.valeur(x), self.historique


# ----------------------------- Saisie utilisateur interactive -----------------------------
def read_float(prompt):
    while True:
        raw = input(prompt).strip().replace(",", ".")
        try:
            return finite_float(raw, prompt)
        except ValidationError as e:
            print(f"{Style.FAIL}{e}{Style.END}")

def read_int(prompt, min_val, max_val, default=None):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            v = int(raw)
        except ValueError:
            print(f"{Style.FAIL}Entier requis.{Style.END}")
            continue
        if min_val <= v <= max_val:
            return v
        print(f"{Style.FAIL}Valeur entre {min_val} et {max_val}.{Style.END}")

def read_constraints():
    section("Saisie des contraintes")
    print("Chaque contrainte : α₁·x₁ + α₂·x₂ + β ≥ 0")
    print("Si vous avez une inégalité '≤', choisissez le sens '≤' et le programme convertira.\n")
    nb = read_int("Nombre de contraintes", 0, 50, default=0)
    constraints = []
    for i in range(1, nb+1):
        print(f"\n{Style.BOLD}Contrainte h{i}{Style.END}")
        a1 = read_float("  α₁ (coeff de x1) : ")
        a2 = read_float("  α₂ (coeff de x2) : ")
        b = read_float("  β (terme constant) : ")
        print("  Sens :")
        print("    1 : ≥ 0")
        print("    2 : ≤ 0")
        sens = read_int("  Choix", 1, 2, default=1)
        if sens == 2:
            a1, a2, b = -a1, -a2, -b
            print(f"  → Convertie en : {a1}·x1 + {a2}·x2 + {b} ≥ 0")
        try:
            constraints.append(Contrainte(i, np.array([a1, a2]), b))
        except ValidationError as e:
            print(f"{Style.FAIL}Erreur : {e}{Style.END} - recommencez cette contrainte.")
            continue
    return constraints

def read_start_point(constraints, tol):
    section("Point de départ")
    while True:
        x1 = read_float("x1 initial : ")
        x2 = read_float("x2 initial : ")
        x0 = np.array([x1, x2])
        violated = [(c, c.valeur(x0)) for c in constraints if c.valeur(x0) < -tol]
        if not violated:
            return x0
        print(f"{Style.FAIL}Point non admissible. Contraintes violées :{Style.END}")
        for c, val in violated:
            print(f"  {c.texte()} → valeur = {format_decimal(val)}")
        print("Veuillez ressaisir.\n")

def interactive_problem():
    print(f"{Style.BOLD}\nGRADIENT PROJETÉ - INTERACTIF{Style.END}")
    print("Minimisation de f(x1,x2) = a·x1² + b·x2² + c·x1 + d·x2\n")
    section("Fonction objectif")
    while True:
        a = read_float("Coefficient a (x1²) : ")
        b = read_float("Coefficient b (x2²) : ")
        c = read_float("Coefficient c (x1) : ")
        d = read_float("Coefficient d (x2) : ")
        try:
            prob = ProblemeQuadratique(a, b, c, d)
            break
        except ValidationError as e:
            print(f"{Style.FAIL}{e}{Style.END}")
    constraints = read_constraints()
    x0 = read_start_point(constraints, EPSILON)
    max_iter = read_int("Nombre maximal d'itérations", 1, 1000, default=MAX_ITER_DEFAULT)
    return prob, constraints, x0, max_iter, EPSILON

def main():
    try:
        prob, constraints, x0, max_iter, tol = interactive_problem()
        solver = GradientProjete(prob, constraints, x0, max_iter, tol)
        x_opt, f_opt, _ = solver.resoudre(verbose=True)
        print(f"\n{Style.GREEN}{Style.BOLD}✓ Résolution réussie.{Style.END}")
        return 0
    except ResolutionError as e:
        print(f"\n{Style.FAIL}{Style.BOLD}Erreur :{Style.END} {e}")
        return 2
    except KeyboardInterrupt:
        print(f"\n{Style.WARNING}Interruption utilisateur.{Style.END}")
        return 130
    except Exception as e:
        print(f"\n{Style.FAIL}{Style.BOLD}Erreur inattendue :{Style.END}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())