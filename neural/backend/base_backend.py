"""
Backend Classes For Model

Backend Classes can define a variety of behaviors for the Models,
including (but not limited to):

1. Compilation
2. Resetting
3. Updating
4. Profiling
"""
import typing as tp
import numpy as np
import numpy.typing as npt
import sympy as sp
from warnings import warn
from scipy.integrate import solve_ivp, OdeSolver, odeint
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

from .. import types as tpe
from .. import errors as err

class Backend:
    """MixIn Backend Class"""

    def compile(self):
        """Compile Model ODE
        
        This is an optional method of the child classes
        """
        raise NotImplementedError

    def clip(self, states: dict = None) -> None:
        """Clip the State Variables

        Clip the state variables in-place after calling the numerical solver.

        The state variables are usually bounded, for example, binding
        variables are bounded between 0 and 1. However, numerical sovlers
        might cause the value of state variables exceed its bounds. A hard
        clip is forced here to ensure the state variables remain in the
        given bounds.
        """
        states = self.states if states is None else states
        for var, (lb, ub) in self.bounds.items():
            states[var].clip(lb, ub, out=states[var])


    def step(self, d_t: float, **input_args) -> None:
        """Step integration once"""
        self._step_euler(d_t, **input_args)

    def _step_euler(self, d_t: float, **input_args) -> None:
        """Euler's method"""
        self.ode(**input_args)
        for var, grad in self.gstates.items():
            self.states[var] += d_t * grad

    def reset(self) -> None:
        """Reset Model
        
        Sets states to initial values, and sets gstates to 0.
        """
        for key, val in self.initial_states.items():
            if np.isscalar(val):
                self.states[key] = val
            else:
                self.states[key].fill(val)
        for key, val in self.gstates.items():
            if np.isscalar(val):
                self.gstates[key] = 0.
            else:
                self.gstates[key].fill(0.)

    @property
    def states_arr(self) -> np.ndarray:
        """State Vector for Batched ODE Solver

        This attribute stackes all state values into a
        :code:`(len(self.states), self.num)` shaped array. It is done for ODE
        solver to handle the state variable easily.
        """
        return np.vstack(list(self.states.values()))

    @states_arr.setter
    def states_arr(self, new_value) -> None:
        """Settting state_vector set states dictionary

        The setter and getter for states_arr is intended to ensure consistency
        between `self.states` and `self.states_arr`
        """
        for var_name, new_val in zip(self.states.keys(), new_value):
            self.states[var_name] = new_val

    def solve(self,
        t: np.ndarray,
        *,
        solver: tpe.SupportedBackend = None,
        reset: bool = True,
        verbose: tp.Union[bool, str] = True,
        callbacks: tp.Union[tp.Callable, tp.Iterable[tp.Callable]] = None,
        solver_kws: tp.Mapping[str, tp.Any] = None,
        **stimuli,
    ) -> tp.Union[tp.Dict, tp.Tuple[tp.Dict, tp.Any]]:
        """Solve model equation for entire input

        Positional Arguments:
            t: 1d numpy array of time vector of the simulation

        Keyword-Only Arguments:
            solver: Which ODE solver to use, defaults to the first entry in the
              :code:`Supported_Solvers` attribute.

                - `Euler`: Custom forward euler solver, default.
                - `odeint`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `LSODA`: Use :py:mod:`scipy.integrate.odeint` which uses LSODA
                - `RK45/RK23/DOP853`: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the specified method
                - :py:mod:`scipy.integrate.OdeSolver` instance: Use
                  :py:mod:`scipy.integrate.solve_ivp` with the provided custom solver

            reset: whether to reset the initial state value of the
              model to the values in :code:`Default_State`. Default to True.
            verbose: If is not `False` a progress bar will be created. If is `str`,
              the value will be set to the description of the progress bar.
            full_output: whether to return the entire output from scipy's
              ode solvers.
            callbacks: functions of the signature :code:`function(self)` that is
              executed for :code:`solver=Euler` at every step.
            solver_kws: a dictionary containingarguments to be passed into the ode
              solvers if scipy solvers are used.

                .. seealso: :py:mod:`scipy.integrate.solve_ivp` and
                    :py:mod:`scipy.integrate.odeint`

        .. note::

            String names for :code:`solve_ivp` (RK45/RK23/DOP853)
            are case-sensitive but not for any other methods.
            Also note that the solvers can hang if the amplitude scale of
            :code:`I_ext` is too large.


        Keyword Arguments:
            stimuli: Key value pair of input arguments that matches the signature
              of the :func:`ode` function.

        Returns:
            Return dictionary of a 2-tuple depending on argument
            :code:`full_output`:

            - `False`: An dictionary of simulation results keyed by state
              variables and each entry is of shape :code:`(num, len(t))`
            - `True`: A 2-tuple where the first entry is as above, and the
              second entry is the ode result from either
              :py:mod:`scipy.integrate.odeint` or
              :py:mod:`scipy.integrate.solve_ivp`. The second entry will be
              :code:`None` if solver is :code:`Euler`
        """
        # validate solver
        if solver is None:
            solver = self.Solvers[0]
        if not isinstance(solver, OdeSolver) and solver not in self.Solvers:
            raise err.NeuralModelError(
                f"Solver '{solver}' not understood, must be one of "
                f"{self.Solvers}."
            )
        solver_kws = {} if solver_kws is None else solver_kws

        # Validate Stimuli 
        # check to make sure that the keyword arguments contain only
        # arguments that are relevant to the model input.
        if stimuli:
            if _extraneous_input_args := set(stimuli.keys()) - set(self._input_args):
                 raise err.CompNeuroModelError(
                    (
                        f"Extraneous input arguments '{_extraneous_input_args}' "
                        "treated as stimuli but are not found in the function "
                        f"definition of {self.__class__.__name__}.ode(), "
                        f"the only supported input variables are '{self._input_args}'"
                    )
                )
            if _missing_input_args := set(self._input_args) - set(stimuli.keys()):
                raise err.CompNeuroModelError(
                    f"Input argument '{_missing_input_args}' missing but are required "
                    f"by the {self.__class__.__name__}.ode() method. Please provide "
                    f"all inputs in '{self._input_args}'."
                )

        # whether to reset initial state to `self.initial_states`
        if reset:
            self.reset()

        # rescale time axis appropriately
        t_long = t * self.Time_Scale
        state_var_shape = self.states_arr.shape
        x0 = np.ravel(self.states_arr)
        d_t = t_long[1] - t_long[0]

        # check external current dimension. It has to be either 1D array the
        # same shape as `t`, or a 2D array of shape `(len(t), self.num)`
        if stimuli:
            for var_name, stim in stimuli.items():
                if stim.ndim == 1:
                    stim = np.repeat(stim[:, None], self.num, axis=-1)
                elif stim.ndim != 2:
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' must be 1D or 2D array"
                    )
                if len(stim) != len(t):
                    raise err.CompNeuroModelError(
                        f"Stimulus '{var_name}' first dimesion must be the same length as t"
                    )
                if stim.shape[1] > 1:
                    if stim.shape != (len(t_long), self.num):
                        raise err.CompNeuroModelError(
                            f"Stimulus '{var_name}' expects shape ({len(t_long)},{self.num}), "
                            f"got {stim.shape}"
                        )
                stimuli[var_name] = stim

        # Register callback that is executed after every euler step.
        callbacks = [] if callbacks is None else np.atleast_1d(callbacks).tolist()
        for f in callbacks:
            if not callable(f):
                raise err.CompNeuroModelError("Callback is not callable\n" f"{f}")
        callbacks = tuple(list(self.callbacks) + callbacks)
        
        # Solve 
        res = np.zeros((len(t_long), len(self.states_arr), self.num))
        # run loop
        iters = enumerate(t_long)
        if verbose:
            iters = tqdm(
                iters,
                total=len(t_long),
                desc=verbose if isinstance(verbose, str) else "",
                dynamic_ncols=True
            )

        for tt, _t in iters:
            _stim = {var_name: stim[tt] for var_name, stim in stimuli.items()}
            self.update()
            d_x = np.vstack(self.ode(_t, self.states_arr, **_stim))
            self.states_arr += d_t * d_x
            self.clip()
            res[tt] = self.states_arr
            if callbacks is not None:
                for _func in callbacks:
                    _func(self)
        # move time axis to last so that we end up with shape
        # (len(self.states), self.num, len(t))
        res = np.moveaxis(res, 0, 2)
        res = {key: res[n] for n, key in enumerate(self.states.keys())}

        return res

        # # Solve IVP Methods
        # if verbose:
        #     pbar = tqdm(
        #         total=len(t_long), desc=verbose if isinstance(verbose, str) else ""
        #     )

        # # 1. create update function for IVP
        # jacc_f = None
        # if stimuli:  # has external input
        #     interpolators = {
        #         var_name: interp1d(
        #             t_long, stim, axis=0, kind="zero", fill_value="extrapolate"
        #         )
        #         for var_name, stim in stimuli.items()
        #     }
        #     if self.jacobian is not None:
        #         # rewrite jacobian function to include invaluation at input value
        #         def jacc_f(t, states):  # pylint:disable=function-redefined
        #             return self.jacobian(  # pylint:disable=not-callable
        #                 t,
        #                 states,
        #                 **{var: intp_f(t) for var, intp_f in interpolators.items()},
        #             )

        #     # the update function interpolates the value of input at every
        #     # step `t`
        #     def update(t_eval, states):
        #         if verbose:
        #             pbar.n = int((t_eval - t_long[0]) // d_t)
        #             pbar.update()
        #         d_states = np.vstack(
        #             self.ode(
        #                 t=t_eval,
        #                 states=np.reshape(states, state_var_shape),
        #                 **{
        #                     var: intp_f(t_eval) for var, intp_f in interpolators.items()
        #                 },
        #             )
        #         )
        #         return d_states.ravel()

        # else:  # no external input
        #     jacc_f = self.jacobian

        #     # if no current is provided, the system solves ode as defined
        #     def update(t_eval, states):
        #         if verbose:
        #             pbar.n = int((t_eval - t_long[0]) // d_t)
        #             pbar.update()
        #         d_states = np.vstack(
        #             self.ode(states=np.reshape(states, state_var_shape), t=t_eval)
        #         )
        #         return d_states.ravel()

        # # solver system
        # ode_res_info = None
        # res = np.zeros((len(t_long), len(self.states_arr), self.num))
        # if isinstance(solver, OdeSolver):
        #     rtol = solver_kws.pop("rtol", 1e-8)
        #     ode_res = solve_ivp(
        #         update,
        #         t_span=(t_long.min(), t_long.max()),
        #         y0=x0,
        #         t_eval=t_long,
        #         method=solver,
        #         rtol=rtol,
        #         jac=jacc_f,
        #     )
        #     ode_res_info = ode_res
        #     res = ode_res.y.reshape((len(self.states_arr), -1, len(t_long)))
        # elif solver.lower() in ["lsoda", "odeint"]:
        #     ode_res = odeint(
        #         update,
        #         y0=x0,
        #         t=t_long,
        #         tfirst=True,
        #         full_output=full_output,
        #         Dfun=jacc_f,
        #         **solver_kws,
        #     )
        #     if full_output:
        #         ode_res_y = ode_res[0]
        #         ode_res_info = ode_res[1]
        #         res = ode_res_y.T.reshape((len(self.states_arr), -1, len(t_long)))
        #     else:
        #         res = ode_res.T.reshape((len(self.states_arr), -1, len(t_long)))
        # else:  # any IVP solver
        #     rtol = solver_kws.pop("rtol", 1e-8)
        #     options = {"rtol": rtol}
        #     if solver.lower() in IVP_SOLVER_WITH_JACC:
        #         options["jac"] = jacc_f
        #     ode_res = solve_ivp(
        #         update,
        #         t_span=(t_long.min(), t_long.max()),
        #         y0=x0,
        #         t_eval=t_long,
        #         method=solver,
        #         **options,
        #     )
        #     ode_res_info = ode_res
        #     res = ode_res.y.reshape((len(self.states_arr), -1, len(t_long)))

        # res = {key: res[n] for n, key in enumerate(self.states.keys())}

        # if verbose:
        #     pbar.update()
        #     pbar.close()

        # if full_output:
        #     return res, ode_res_info
        # return res