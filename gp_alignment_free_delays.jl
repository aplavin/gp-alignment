# GP-based alignment of multi-slice light curves with free per-slice delays.
#
# Given flux-density light curves measured at different radial offsets r from a
# VLBI core, this script fits a shared latent light-curve f(t) (Gaussian process)
# and per-slice parameters (time delay τ_i, amplitude a_i, baseline b_i) so that:
#
#   logflux_i(t) ≈ a_i · f(t - τ_i) + b_i
#
# The first slice is the reference: τ_1 = 0, a_1 = 1.
# Fitting is MAP estimation (L-BFGS), with uncertainties from a Laplace approximation.

using DataManipulation
using QuackIO, StructArrays
using Uncertain
using LinearAlgebra
using KernelFunctions: Matern52Kernel, with_lengthscale, kernelmatrix
using AbstractGPs
using Interpolations: linear_interpolation
using Optimization, OptimizationOptimJL, ForwardDiff
using IntervalSets
using MakieExtra; import CairoMakie

# ============================================================
# Configuration
# ============================================================

# GP hyperparameters
const σ_GP = 1.0          # prior amplitude
const ℓ_GP = 1.0          # lengthscale (years)

# Discretisation grid for the latent function f(t)
const M = 150             # number of grid points
const t_start = 2006.0
const t_end   = 2028.0

# Data was pre-processed with the following settings (baked into the CSV):
#   beam_fwhm = 0.7 mas, n_radii ≈ 28, t_min = 2010.0
#   err = (0.05·flux + 0.005) · corr_factor   (corr_factor ≈ 2.3, there are multiple slices per beam)

# ============================================================
# Load data
# ============================================================

@info "Loading data..."
raw = QuackIO.read_csv(StructArray, joinpath(@__DIR__, "data.csv"))

slices = @p raw |>
    mapinsert(logflux_u = log(U.Value(_.flux, _.err))) |>
    mapinsert(logflux = _.logflux_u.v, logerr = _.logflux_u.u) |>
    sort(by=_.year) |>
    group_vg(_.r_mas)
@info "Loaded $(length(slices)) slices, $(sum(length, slices)) observations"

# ============================================================
# GP grid & kernel
# ============================================================

const t_grid = range(t_start, t_end, length=M)
const kernel_gp = σ_GP^2 * with_lengthscale(Matern52Kernel(), ℓ_GP)
const K_inv = inv(Symmetric(kernelmatrix(kernel_gp, t_grid)))
const gp_obj = GP(kernel_gp)

# ============================================================
# Parameter pack/unpack
# ============================================================
# Optimisation vector layout:
#   x[1:M]       — f values on the GP grid
#   x[M+1]       — b₁ (baseline for reference slice; τ₁=0, a₁=1 fixed)
#   x[M+2 : ...]  — per-slice params for slices 2:N:
#                    [τ₂..τ_N, log_a₂..log_a_N, b₂..b_N]

function pack_params(f, b₁, slice_params)
    vcat(f, [b₁], reduce(vcat, StructArrays.components(slice_params)))
end

function unpack_params(x, n_rest)
    f  = @view x[1:M]
    b₁ = x[M + 1]
    offset = M + 1
    slice_params = StructArray(
        τ     = @view(x[offset+1         : offset+n_rest]),
        log_a = @view(x[offset+n_rest+1  : offset+2n_rest]),
        b     = @view(x[offset+2n_rest+1 : offset+3n_rest]),
    )
    (; f, b₁, slice_params)
end

# ============================================================
# Model: negative log-posterior
# ============================================================

function neg_logpost(x, data)
    (; slices, K_inv) = data
    (; f, b₁, slice_params) = unpack_params(x, length(slices) - 1)

    # GP prior: ½ fᵀ K⁻¹ f
    gp_prior = 0.5 * dot(f, K_inv * f)

    itp = linear_interpolation(t_grid, f)

    # Slice 1 (reference): a₁ = 1, τ₁ = 0
    data_fit = sum(slices[1]) do obs
        resid = obs.logflux - (itp(obs.year) + b₁)
        resid^2 / obs.logerr^2
    end

    # Slices 2:N
    for (s, sp) in zip(@view(slices[2:end]), slice_params)
        a_i = exp(sp.log_a)
        data_fit += sum(s) do obs
            resid = obs.logflux - (a_i * itp(obs.year - sp.τ) + sp.b)
            resid^2 / obs.logerr^2
        end
    end

    # Negative log-posterior = ½ χ² + ½ fᵀK⁻¹f  (up to a constant).
    # The ½ comes from the Gaussian log-likelihood: -ln N(x|μ,σ²) = ½(x-μ)²/σ² + const.
    return data_fit / 2 + gp_prior
end

# ============================================================
# Fit
# ============================================================

# Initialisation
slice_params_init = StructArray(
    τ     = (@p slices[2:end] map(1.0 * key(_))),                   # τ ≈ 1.0 * r (guess: ~1 yr/mas)
    log_a = zeros(length(slices) - 1),                              # a = 1
    b     = (@p slices[2:end] map(sum(_.logflux) / length(_))),     # mean log-flux
)
b₁_init = sum(slices[1].logflux) / length(slices[1])
x0 = pack_params(zeros(M), b₁_init, slice_params_init)
@info "Fitting: $(length(slices)) slices, $(length(x0)) parameters..."

opt_data = (; slices, K_inv)
optf = OptimizationFunction(neg_logpost, AutoForwardDiff())
prob = OptimizationProblem(optf, x0, opt_data)
sol  = solve(prob, LBFGS(); maxiters=500)
@info "Optimisation done" retcode=sol.retcode objective=sol.objective

# Extract fitted parameters into a single StructArray (one row per slice)
fit = let
    (; f, b₁, slice_params) = unpack_params(sol.u, length(slices) - 1)
    StructArray(
        r   = (@p slices map(key(_))),
        τ   = [0.0; slice_params.τ],
        a   = [1.0; exp.(slice_params.log_a)],
        b   = [b₁;  slice_params.b],
    )
end

# GP posterior (exact conditioning via AbstractGPs)
aligned = flatmap(zip(slices, fit)) do (s, fp)
    StructArray(
        t   = s.year .- fp.τ,
        y   = (s.logflux .- fp.b) ./ fp.a,
        var = (s.logerr ./ fp.a).^2,
    )
end
post_gp = posterior(gp_obj(aligned.t, Diagonal(aligned.var)), aligned.y)

# Uncertainties: Laplace approximation (Hessian at MAP)
@info "Computing Hessian for uncertainty estimates..."
H = ForwardDiff.hessian(x -> neg_logpost(x, opt_data), sol.u) |> Symmetric
any(e -> e < 0, eigvals(H)) && @warn "Hessian has negative eigenvalues — solution may not be at a minimum"
Σ_full = inv(H)
std_all = unpack_params(.√(diag(Σ_full)), length(slices) - 1)
fit = @insert fit.τ_σ = [0.0; std_all.slice_params.τ]
fit = @insert fit.a_σ = [0.0; fit.a[2:end] .* std_all.slice_params.log_a]   # δa = a · δ(log a)
fit = @insert fit.b_σ = [std_all.b₁; std_all.slice_params.b]

# ============================================================
# Plot: summary
# ============================================================

@info "Plotting..."

fig = Figure(size=(1200, 900))

# Left column: three panels sharing x = offset from core
ax_τ = Axis(fig[1, 1]; ylabel="Time delay τ (yr)", xticklabelsvisible=false)
band!(ax_τ, fit.r, fit.τ .± fit.τ_σ; color=(:steelblue, 0.2))
scatterlines!(ax_τ, fit.r, fit.τ; color=:steelblue, linewidth=2, markersize=5)

ax_a = Axis(fig[2, 1]; ylabel="aᵢ (amplitude scale)", xticklabelsvisible=false)
scatterlines!(ax_a, fit.r, fit.a; color=:steelblue, markersize=6, linewidth=1.5)

ax_b = Axis(fig[3, 1]; xlabel="Offset from core (mas)", ylabel="bᵢ (log-flux offset)")
scatterlines!(ax_b, fit.r, fit.b; color=:steelblue, markersize=6, linewidth=1.5)

linkxaxes!(ax_τ, ax_a, ax_b)

# Right column: aligned data with GP posterior
colormap = :turbo
colorrange = (0, maximum(fit.r))
Axis(fig[1:3, 2]; limits=(2012..2027, -2.2..2.2),
     xlabel="Time − τᵢ (year)", ylabel="(logflux − bᵢ) / aᵢ", title="Aligned data + GP posterior")
for (s, fp) in zip(slices, fit)
    scatter!(
        s.year .- fp.τ, (s.logflux .- fp.b) ./ fp.a;
        markersize=4, color=key(s), colormap, colorrange)
end
post_data = let
    t = range(2009.0, 2027.0, length=300)
    f, v = mean_and_var(post_gp(t))
    StructArray(; t, f=U.Value.(f, .√(v)))
end
multiplot!((lines, band => (;alpha=0.15)), post_data.t, post_data.f;
           color=:black, linewidth=2)

Colorbar(fig[1:3, 3]; colormap, colorrange, label="r (mas)")

colsize!(fig.layout, 2, Relative(2/3))
resize_to_layout!(fig)
save(joinpath(@__DIR__, "gp_alignment_summary.png"), fig)
@info "Saved gp_alignment_summary.png"

# ============================================================
# Save results
# ============================================================

QuackIO.write_table(joinpath(@__DIR__, "fitted_slice_parameters.csv"), fit)
@info "Saved fitted_slice_parameters.csv (objective = $(round(sol.objective, digits=2)))"

gp_post = StructArray(year=post_data.t, mean=U.value.(post_data.f), std=U.uncertainty.(post_data.f))
QuackIO.write_table(joinpath(@__DIR__, "gp_posterior_mean_std.csv"), gp_post)
@info "Saved gp_posterior_mean_std.csv"
