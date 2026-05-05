---
date: '2026-05-05'
title: 'NBA Score Projector'
github: 'https://github.com/Perseus1729'
external: '/nba-projector/'
tech:
  - JavaScript
  - Python
  - scikit-learn
  - Chart.js
  - Piecewise Regression
showInProjects: true
---

Live, interactive NBA final-score projector that compares 17 regression and piecewise-polynomial models trained on 14 seasons of real NBA team-game data (2010–2024). Pick the home team, choose a history window (recent-5, weighted, full), enter the current score and minute — the page renders every model's prediction side-by-side with held-out MAE/RMSE, plus a multi-line chart of each model's projected trajectory. Pipeline scripts under `/nba-projector/pipeline/` regenerate priors and refit models from the latest season.
