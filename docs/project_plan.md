<!-- \subsection{Какие артефакты считаются результатом}

Для каждого обучающего запуска (run) мы обязаны иметь:
\begin{itemize}
  \item \texttt{metrics.jsonl} (val\_loss/val\_acc по эпохам),
  \item \texttt{checkpoints/final.pt} (минимум, с которым дальше работаем),
  \item \texttt{summary.json} (контекст: seed, время, параметры)
\end{itemize}

Для ``ландшафтных'' экспериментов мы получаем три типа артефактов:
\begin{itemize}
  \item \textbf{Interpolation}: CSV \texttt{interp\_\_*.csv} со столбцами $(t, \mathrm{val\_loss}, \mathrm{val\_acc})$
  \item \textbf{Barrier}: JSON \texttt{barrier\_\_*.json} + агрегат \texttt{barriers.csv}
  \item \textbf{Geometry}: JSON \texttt{geometry\_\_*.json} + агрегат \texttt{geometries.csv}
\end{itemize}

\paragraph{Оговорка про барьер/jump}
Мы не утверждаем, что находим минимальный барьер в пространстве весов
Мы фиксируем \emph{операциональный тест}:

\[
\theta(t) = (1-t)\theta_A + t\theta_B
\quad\text{(linear weight interpolation)}
\]

или его piecewise-вариант (polyline через pivot). Барьер $\Delta L$ считается \emph{строго относительно выбранного пути} и определения baseline (например \texttt{max\_minus\_linear\_baseline}) -->

\section{Порядок работы}

\begin{enumerate}
  \item \textbf{Baseline-train:} сначала проверяем, что обучение стабильно и даёт адекватный val\_acc
  \item \textbf{Baseline-landscape:} берём пару независимых минимумов в одном режиме и измеряем интерполяцию/барьер/геометрию. Это ``нулевая точка''
  \item \textbf{Noise grid:} систематически меняем режим шума (batch/lr), повторяем весь цикл, собираем таблицы, сравниваем
  \item \textbf{(Опц) Polyline paths:} если линейный путь даёт барьер, проверяем, снижается ли он через pivot(ы)
\end{enumerate}

\section{Эксперимент 1 --- Baseline обучение}

\subsection*{Задача}
Получить устойчивые чекпоинты минимумов и опорное качество ResNet18 на CIFAR-10

\subsection*{Конфигурация}
\texttt{configs/train/sgd\_base.yaml}
Количество seed: 5

\subsection*{Запуски}
Для каждого seed $s$:
\begin{verbatim}
ntempvh train --config configs/train/sgd_base.yaml --seed s --out outputs/runs
\end{verbatim}

\subsection*{Минимальные sanity checks}
\begin{itemize}
  \item \textbf{val\_acc} заметно больше $0.1$ (иначе это ``угадывание'' CIFAR-10)
  \item В логах нет NaN/inf и нет явной дивергенции loss
  \item Между seed есть \emph{разброс}, но не хаос: качество в одном диапазоне
\end{itemize}

\section{Эксперимент 2 --- Linear interpolation и barrier (mode connectivity proxy)}

\subsection*{Что проверяем}
Насколько ``соединены'' два независимых минимума, найденные одинаковым режимом обучения.
Операционально: строим $L(t)$ вдоль линейного пути и измеряем $\Delta L$

\subsection*{Выбор пары минимумов}
Берём два run с одинаковыми гиперпараметрами и разными seed:
\[
A=\texttt{.../seed s\_0/.../final.pt},\qquad
B=\texttt{.../seed s\_1/.../final.pt}.
\]

\subsection*{Интерполяция}
Конфиг: \texttt{configs/eval/interpolation.yaml}
<!-- Критические поля:
\begin{itemize}
  \item \texttt{path.type = linear}
  \item \texttt{path.num\_points} (например 41)
  \item \texttt{path.bn\_recalib\_batches > 0} (иначе BatchNorm ломает кривую)
  \item \texttt{evaluation.batch\_size} (batch для оценки)
\end{itemize} -->

Команда:
\begin{verbatim}
ntempvh interpolate --ckptA A --ckptB B --config configs/eval/interpolation.yaml
\end{verbatim}

\subsection*{Barrier}
Конфиг: \texttt{configs/eval/barrier.yaml}, 
<!-- например
\texttt{definition = max\_minus\_linear\_baseline}. -->

Команда:
\begin{verbatim}
ntempvh barrier --interp_csv outputs/artifacts/interp/<interp.csv> \
  --config configs/eval/barrier.yaml
\end{verbatim}

\subsection*{Как интерпретируем}
\begin{itemize}
  \item Большой $\Delta L$ --- выраженный ``барьер'' по выбранному пути
  \item Малый $\Delta L$ --- практически ``no barrier'' для линейной интерполяции
  \item Смотрим не только $\Delta L$, но и $\Delta L_{\mathrm{rel}}$ (нормировка уже добавлены в json)
\end{itemize}

\section{Эксперимент 3 --- Geometry (локальная кривизна около минимума)}

\subsection*{Что проверяем}
Прокси-показатель ``плоскости/кривизны'' минимума через центральную разность второго порядка по случайным направлениям.

\subsection*{Запуск}
Конфиг: \texttt{configs/eval/geometry.yaml}.
Поля:
\begin{itemize}
  \item \texttt{alpha} задаёт $\varepsilon=\alpha\|\theta\|$,
  \item \texttt{num\_directions} (точность оценки),
  \item \texttt{eval\_batch\_size} (ускорение и стабильность оценки).
\end{itemize}

Для чекпоинтов $A$ и $B$:
\begin{verbatim}
ntempvh geometry --ckpt A --config configs/eval/geometry.yaml
ntempvh geometry --ckpt B --config configs/eval/geometry.yaml
\end{verbatim}

\subsection*{Как интерпретируем}
\begin{itemize}
  \item Меньшее $\kappa_{\mathrm{tr}}$ \;\;$\Rightarrow$\;\; более ``плоский'' минимум (в смысле выбранного прокси)
  \item Значение зависит от $\alpha$ и от того, на каких данных оцениваем loss (val сплит)
\end{itemize}

\section{Эксперимент 4 --- Noise grid (batch/lr) + полный цикл}

\subsection*{Идея}
Меняем режим шума SGD и проверяем, как это отражается одновременно на:
качество $\leftrightarrow$ барьеры $\leftrightarrow$ геометрия.

\subsection*{Сетка параметров}
Минимум:
\[
\mathrm{batch}\in\{64,128,256\},\quad
\mathrm{lr}\in\{0.05,0.1,0.2\},\quad
\mathrm{seed}\in\{0,1\}.
\]

\subsection*{Протокол для одной точки сетки $(\mathrm{batch},\mathrm{lr})$}
\begin{enumerate}
  \item Два независимых обучения (seed 0 и 1) $\Rightarrow$ два \texttt{final.pt}
  \item Интерполяция между ними $\Rightarrow$ \texttt{interp.csv}
  \item Barrier по \texttt{interp.csv} $\Rightarrow$ \texttt{barrier.json} и строка в \texttt{barriers.csv}
  \item Geometry хотя бы для одного чекпоинта $\Rightarrow$ \texttt{geometry.json} и строка в \texttt{geometries.csv}
\end{enumerate}

\subsection*{Что в итоге сравниваем}
Для каждого режима:
\begin{itemize}
  \item финальный val\_acc (и/или лучший по val\_loss),
  \item $\Delta L$ и $\Delta L_{\mathrm{rel}}$,
  \item $\kappa_{\mathrm{tr}}$
\end{itemize}

Дальше строим уже простые зависимости (таблично/графиками):
$\Delta L$ vs (batch,lr), $\kappa$ vs (batch,lr), val\_acc vs $\Delta L$, val\_acc vs $\kappa$

\section{Эксперимент 5 --- Piecewise-linear (pivot) пути}

\subsection*{Зачем}
Если линейный путь показывает барьер, мы проверяем, что барьер \emph{можно снизить} полилинией через pivot
Это демонстрация того, что ``барьер'' не обязательно фундаментальный, а может быть артефактом выбранного пути

\subsection*{Настройки}
В \texttt{configs/eval/interpolation.yaml}:
\begin{verbatim}
path:
  type: piecewise_linear
  num_points: 41
  bn_recalib_batches: 20
  pivots:
    - /path/to/pivot1.pt
\end{verbatim}

\subsection*{Запуск}
\begin{verbatim}
ntempvh interpolate --ckptA A --ckptB B --config configs/eval/interpolation.yaml
ntempvh barrier --interp_csv outputs/artifacts/interp/<interp_piecewise.csv> \
  --config configs/eval/barrier.yaml
\end{verbatim}

\subsection*{Критерий}
Сравниваем $\Delta L_{\text{linear}}$ и $\Delta L_{\text{piecewise}}$ на одинаковой паре $(A,B)$