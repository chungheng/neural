class CUDABackend(Backend):
    def __init__(self, model, **kwargs):
        self.backend = kwargs.pop("backend", None)
        self.num = kwargs.pop("num", None)
        self.data = dict()
        self.model = model
        self.dtype = kwargs.pop("dtype", np.float64)
        self.compile(**kwargs)

        self._clip = ElementwiseKernel(
            "float *value, float a_min, float a_max",
            "value[i] = value[i] > a_min ? (value[i] < a_max ? value[i] : a_max) : a_min",
            "clip",
        )

    def _allocate_cuda_memory(self, key):
        """
        allocate GPU memroy for variable
        """
        if key in self.data and len(self.data[key]) != self.num:
            del self.data[key]

        if key not in self.data:
            array = garray.empty(self.num, dtype=self.dtype)
            self.data[key] = array

    def reset(self, **kwargs):
        """
        reset the gpu data.

        Reset the GPU data to default values.

        Arguments:
            kwargs (dict): keyward arguments.
        """
        params = []
        items = chain(self.model.states.items(), self.model.params.items())
        for key, val in items:
            val = kwargs.pop(key, val)

            # allocate GPU memory
            if key in self.model.states:
                self._allocate_cuda_memory(key)
            elif hasattr(val, "__len__"):  # params with __len__
                self._allocate_cuda_memory(key)
                params.append(key)

            if isinstance(val, np.ndarray):
                if val.dtype != self.dtype:
                    val = val.astype(self.dtype)
                drv.memcpy_htod(self.data[key].gpudata, val)
            elif isinstance(val, garray.GPUArray):
                if key in self.model.params:
                    if val.dtype != self.dtype:
                        raise err.NeuralBackendError(
                            f"Model Param '{key}' has dtype {val.dtype} but Backend expects {self.dtype}"
                        )
                    self.data[key] = val
                    continue
                if val.dtype != self.dtype:
                    val = val.get()
                    val = val.astype(self.dtype)
                    drv.memcpy_htod(self.data[key].gpudata, val)
                else:
                    drv.memcpy_dtod(self.data[key].gpudata, val.gpudata, val.nbytes)
            elif isinstance(val, Number):
                if key in self.model.params:
                    self.model.params[key] = val
                    continue
                self.data[key].fill(val)
            else:
                raise err.NeuralBackendError(f"Invalid variable '{key}': {val}")
        return params

    def compile(self, **kwargs):
        """
        compile the cuda kernel.

        Keyword Arguments:
        """

        # decide the number of threads:
        keys = chain(self.model.states.keys(), self.model.params.keys())
        for key in keys:
            val = getattr(self.model, key)
            val = kwargs.get(key, val)
            if hasattr(val, "__len__"):
                _num = len(val)
                self.num = self.num or _num
                if self.num != _num:
                    raise err.err.NeuralBackendError(
                        f"Mismatch in data size for '{key}'. Expect {self.num}, got {_num}."
                    )
        else:
            if not self.num:
                raise err.NeuralBackendError("Please give the number of models to run")

        # reset gpu data
        params = self.reset(**kwargs)

        # assume the rest of kwargs are input-related
        inputs = kwargs.copy()
        for key in inputs.keys():
            if key not in self.model.Inputs:
                err.NeuralBackendError(f"Unexpected input '{key}'")

        # generate cuda kernel, a.k.a self.cuda.kernel
        self.get_cuda_kernel(inputs_gdata=inputs, params_gdata=params)

        if self.has_random:
            self.seed = drv.mem_alloc(self.num * 48)
            self.init_random_seed.prepared_async_call(
                self.grid, self.block, None, self.num, self.seed
            )

    def update(self, d_t, **kwargs):
        """Update State Variables"""
        st = kwargs.pop("st", None)
        args = []
        for key, dtype in zip(self.args, self.arg_ctype[2:]):
            if key == "seed":
                args.append(self.seed)
                continue

            val = self.data.get(key, None)
            try:
                if val is None:
                    val = kwargs[key]
                if hasattr(val, "__len__"):
                    if dtype != "P":
                        raise err.NeuralBackendError(
                            f"[{self.model}] Expect GPU array but get a scalar input: {key}"
                        )
                    if val.dtype != self.dtype:
                        raise err.NeuralBackendError(
                            f"[{self.model}] GPU array float type mismatches: {key}"
                        )
                else:
                    if dtype == "P":
                        raise err.NeuralBackendError(
                            f"[{self.model}] Expect GPU array but get a scalar input: {key}"
                        )
            except Exception as e:
                raise err.NeuralBackendError(f"{self.model} Error") from e

            args.append(val.gpudata if dtype == "P" else val)

        self.kernel.prepared_async_call(self.grid, self.block, st, self.num, d_t, *args)

    def cuda_profile(self, **kwargs):
        num = kwargs.pop("num", 1000)
        niter = kwargs.pop("niter", 1000)
        dtype = kwargs.pop("dtype", np.float64)

        self.cuda_compile(num=num, dtype=dtype)

        args = {key: garray.empty(num, dtype) for key in self.cuda.args}

        start = drv.Event()
        end = drv.Event()
        secs = 0.0

        for i in range(niter):
            start.record()
            self._cuda_update(0.0, **args)
            end.record()
            end.synchronize()
            secs += start.time_till(end)

        for key in args:
            args[key].gpudata.free()

        name = self.__class__.__name__
        print("Average run time of {}: {} ms".format(name, secs / niter))

    def get_cuda_kernel(self, **kwargs):
        code_generator = CudaKernelGenerator(self.model, dtype=self.dtype, **kwargs)
        code_generator.generate()

        try:
            mod = SourceModule(
                code_generator.cuda_src,
                # options=["--ptxas-options=-v", "--expt-relaxed-constexpr"],
                options=["--expt-relaxed-constexpr"],
                no_extern_c=code_generator.has_random,
            )
            func = mod.get_function(self.model.__class__.__name__)
        except Exception as e:
            lines = code_generator.cuda_src.split("\n")
            num_digits = 1 + int(np.floor(np.log10(len(lines))))
            for index, line in enumerate(lines):
                print("{: >{}}: {}".format(index, num_digits, line))
            raise err.NeuralBackendError("CUDA Kernel Generation Error") from e

        self.src = code_generator.cuda_src
        self.args = code_generator.args
        self.arg_ctype = code_generator.arg_type

        func.prepare(self.arg_ctype)
        self.kernel = func

        self.has_random = code_generator.has_random
        if self.has_random:
            init_random_seed = mod.get_function("generate_seed")
            init_random_seed.prepare(code_generator.init_random_seed_arg)
            self.init_random_seed = init_random_seed

        deviceData = pycuda.tools.DeviceData()
        maxThreads = int(deviceData.registers // func.num_regs)
        maxThreads = int(2 ** int(np.log(maxThreads) / np.log(2)))
        threadsPerBlock = int(min(256, maxThreads, deviceData.max_threads))
        self.block = (threadsPerBlock, 1, 1)
        self.grid = (
            int(
                min(
                    6
                    * drv.Context.get_device().MULTIPROCESSOR_COUNT,  # pylint: disable=no-member
                    (self.num - 1) / threadsPerBlock + 1,
                )
            ),
            1,
        )

        return func

    def clip(self, value, a_min, a_max):
        if isinstance(value, garray.GPUArray):
            self._clip(value, a_min, a_max)
        else:
            np.clip(value, a_min, a_max, out=value)
