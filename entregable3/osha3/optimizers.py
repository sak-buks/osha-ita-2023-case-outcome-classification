"""Optimizadores de hiperparámetros con el mismo presupuesto (D5).

Los cuatro llaman a la misma función objetivo ``objective(params) -> (score, info)``, que ajusta
el pipeline completo en cada pliegue interno (sin fugas) y devuelve la balanced accuracy media.
El presupuesto cuenta evaluaciones distintas: una configuración repetida se toma de la caché y
no consume presupuesto.

- Grid: ``ParameterGrid`` de la rejilla del modelo (≤ 30 puntos).
- Random: muestreo uniforme (log-uniforme donde corresponde) del espacio.
- Bayesiano: Optuna con TPE (sustituto: dos densidades l(x) y g(x) sobre las configuraciones
  buenas y malas; adquisición: maximizar l(x)/g(x), equivalente a la mejora esperada).
  Las primeras 10 evaluaciones son aleatorias (``n_startup_trials``).
- Genético: DEAP. Genes en [0, 1] que se decodifican al espacio. Selección por torneo (k=3),
  cruce SBX acotado (eta=15, p=0.7), mutación polinómica acotada (eta=20, p=0.3, indpb=1/genes),
  elitismo de 1 individuo, población de 10. Se registra la diversidad por generación.
"""
import json
import math
import random
import time

import numpy as np
import optuna
from deap import base, creator, tools
from sklearn.model_selection import ParameterGrid

from .data import SEED
from .models import Cat, Float, Int
from .pipeline import complete_params

OPTIMIZERS = ('grid', 'random', 'bayes', 'genetic')
POP_SIZE, CXPB, MUTPB, TOURNSIZE = 10, 0.7, 0.3, 3


def _active(space, params):
    return {k: v for k, v in space.items() if v.when is None or params.get(v.when[0]) == v.when[1]}


def _key(params):
    return json.dumps({k: params[k] for k in sorted(params)}, default=str)


def _from_unit(dist, u):
    """Decodifica un gen u ∈ [0, 1] al valor del hiperparámetro."""
    u = min(max(float(u), 0.0), 1.0)
    if isinstance(dist, Cat):
        return dist.choices[min(int(u * len(dist.choices)), len(dist.choices) - 1)]
    lo, hi = dist.low, dist.high
    if dist.log:
        value = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    else:
        value = lo + u * (hi - lo)
    if isinstance(dist, Int):
        return int(min(max(round(value), lo), hi))
    return float(value)


def decode(space, genes):
    """Convierte un vector de genes (orden de ``space``) en hiperparámetros activos."""
    raw = {name: _from_unit(dist, g) for (name, dist), g in zip(space.items(), genes)}
    return {k: raw[k] for k in _active(space, raw)}


class Budget:
    """Lleva la cuenta de evaluaciones distintas, la caché y el historial."""

    def __init__(self, spec, objective, budget):
        self.spec, self.objective, self.budget = spec, objective, budget
        self.cache, self.history = {}, []

    @property
    def done(self):
        return len(self.history) >= self.budget

    def __call__(self, params):
        params = complete_params(self.spec, params)
        key = _key(params)
        if key in self.cache:
            return self.cache[key]
        if self.done:
            return -np.inf
        start = time.perf_counter()
        score, info = self.objective(params)
        self.cache[key] = score
        self.history.append({'eval': len(self.history) + 1, 'params': params, 'score': float(score),
                             'seconds': time.perf_counter() - start, **info})
        return score


def run_grid(spec, budget):
    grid = list(ParameterGrid(spec.grid))
    if len(grid) > budget.budget:
        raise ValueError(f'La rejilla de {spec.name} tiene {len(grid)} puntos y el presupuesto es {budget.budget}; '
                         'truncarla eliminaría valores completos.')
    for params in grid:
        budget(params)
    return {}


def run_random(spec, budget, seed=SEED):
    rng = np.random.default_rng(seed)
    tries = 0
    while not budget.done and tries < 20 * budget.budget:
        tries += 1
        budget(decode(spec.space, rng.random(len(spec.space))))
    return {}


def run_bayes(spec, budget, seed=SEED):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=10)
    study = optuna.create_study(direction='maximize', sampler=sampler)

    def trial_fn(trial):
        params = {}
        for name, dist in spec.space.items():
            if dist.when is not None and params.get(dist.when[0]) != dist.when[1]:
                continue
            if isinstance(dist, Cat):
                idx = trial.suggest_categorical(name, list(range(len(dist.choices))))
                params[name] = dist.choices[idx]
            elif isinstance(dist, Int):
                params[name] = trial.suggest_int(name, dist.low, dist.high, log=dist.log)
            else:
                params[name] = trial.suggest_float(name, dist.low, dist.high, log=dist.log)
        return budget(params)

    trials = 0
    while not budget.done and trials < 3 * budget.budget:
        study.optimize(trial_fn, n_trials=1)
        trials += 1
    return {'optuna_trials': trials, 'sampler': 'TPESampler(n_startup_trials=10)'}


def _diversity(population):
    genes = np.asarray([list(ind) for ind in population])
    n = len(genes)
    d = [np.linalg.norm(genes[i] - genes[j]) for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d) / math.sqrt(genes.shape[1])) if d else 0.0


def run_genetic(spec, budget, seed=SEED):
    random.seed(seed)
    n_genes = len(spec.space)
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Individual', list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register('individual', tools.initRepeat, creator.Individual, random.random, n_genes)
    tb.register('population', tools.initRepeat, list, tb.individual)
    tb.register('mate', tools.cxSimulatedBinaryBounded, eta=15.0, low=0.0, up=1.0)
    tb.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0 / n_genes)
    tb.register('select', tools.selTournament, tournsize=TOURNSIZE)

    def evaluate(ind):
        return (budget(decode(spec.space, ind)),)

    pop = tb.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = evaluate(ind)
    generations = [{'generation': 0, 'diversity': _diversity(pop), 'best': max(i.fitness.values[0] for i in pop),
                    'evaluations': len(budget.history)}]
    gen = 0
    while not budget.done and gen < 50:
        gen += 1
        elite = tools.selBest(pop, 1)[0]
        offspring = [tb.clone(i) for i in tb.select(pop, len(pop) - 1)]
        for a, b in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                tb.mate(a, b)
                del a.fitness.values, b.fitness.values
        for ind in offspring:
            if random.random() < MUTPB:
                tb.mutate(ind)
                del ind.fitness.values
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        pop = [tb.clone(elite)] + offspring
        generations.append({'generation': gen, 'diversity': _diversity(pop),
                            'best': max(i.fitness.values[0] for i in pop), 'evaluations': len(budget.history)})
    return {'generations': generations, 'population': POP_SIZE, 'cxpb': CXPB, 'mutpb': MUTPB,
            'tournsize': TOURNSIZE, 'elitism': 1}


RUNNERS = {'grid': run_grid, 'random': run_random, 'bayes': run_bayes, 'genetic': run_genetic}


def optimize(name, spec, objective, n_evals):
    """Ejecuta un optimizador y devuelve (historial, mejor configuración, detalles)."""
    budget = Budget(spec, objective, n_evals)
    details = RUNNERS[name](spec, budget)
    best = max(budget.history, key=lambda r: r['score'])
    return budget.history, best, details
