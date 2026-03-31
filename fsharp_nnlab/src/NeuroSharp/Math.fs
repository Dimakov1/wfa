namespace NeuroSharp

module Math =
    let inline sqr x = x * x
    let clamp low high value = max low (min high value)

    let random = System.Random(42)

    let createMatrix rows cols f =
        Array2D.init rows cols f

    let zeros rows cols =
        Array2D.zeroCreate<float> rows cols

    let zerosVector n =
        Array.zeroCreate<float> n

    let copyMatrix (m: float[,]) =
        Array2D.init (Array2D.length1 m) (Array2D.length2 m) (fun r c -> m[r, c])

    let copyVector (v: float[]) = Array.copy v

    let mapMatrix f (m: float[,]) =
        Array2D.init (Array2D.length1 m) (Array2D.length2 m) (fun r c -> f m[r, c])

    let mapVector f (v: float[]) = v |> Array.map f

    let rows (m: float[,]) = Array2D.length1 m
    let cols (m: float[,]) = Array2D.length2 m

    let transpose (m: float[,]) =
        createMatrix (cols m) (rows m) (fun r c -> m[c, r])

    let dot (a: float[,]) (b: float[,]) =
        let aRows = rows a
        let aCols = cols a
        let bRows = rows b
        let bCols = cols b
        if aCols <> bRows then
            invalidArg "b" $"Dimension mismatch in dot: ({aRows}x{aCols}) x ({bRows}x{bCols})"

        createMatrix aRows bCols (fun r c ->
            let mutable acc = 0.0
            for k in 0 .. aCols - 1 do
                acc <- acc + a[r, k] * b[k, c]
            acc)

    let addRowVector (m: float[,]) (v: float[]) =
        if cols m <> v.Length then
            invalidArg "v" "Vector length must equal the number of matrix columns."
        createMatrix (rows m) (cols m) (fun r c -> m[r, c] + v[c])

    let subtract (a: float[,]) (b: float[,]) =
        createMatrix (rows a) (cols a) (fun r c -> a[r, c] - b[r, c])

    let add (a: float[,]) (b: float[,]) =
        createMatrix (rows a) (cols a) (fun r c -> a[r, c] + b[r, c])

    let hadamard (a: float[,]) (b: float[,]) =
        createMatrix (rows a) (cols a) (fun r c -> a[r, c] * b[r, c])

    let scalarMul scalar (m: float[,]) =
        mapMatrix (fun x -> scalar * x) m

    let scalarMulVector scalar (v: float[]) =
        mapVector (fun x -> scalar * x) v

    let sumColumns (m: float[,]) =
        Array.init (cols m) (fun c ->
            let mutable acc = 0.0
            for r in 0 .. rows m - 1 do
                acc <- acc + m[r, c]
            acc)

    let meanColumns (m: float[,]) =
        let batch = float (rows m)
        sumColumns m |> Array.map (fun x -> x / batch)

    let meanSquaredNormMatrix (m: float[,]) =
        let mutable acc = 0.0
        for r in 0 .. rows m - 1 do
            for c in 0 .. cols m - 1 do
                acc <- acc + m[r, c] * m[r, c]
        acc / float (rows m * cols m)

    let clipByNorm maxNorm (m: float[,]) =
        let mutable acc = 0.0
        for r in 0 .. rows m - 1 do
            for c in 0 .. cols m - 1 do
                acc <- acc + m[r, c] * m[r, c]
        let norm = sqrt acc
        if norm <= maxNorm || norm = 0.0 then copyMatrix m
        else
            let scale = maxNorm / norm
            scalarMul scale m

    let clipByNormVector maxNorm (v: float[]) =
        let norm = v |> Array.sumBy (fun x -> x * x) |> sqrt
        if norm <= maxNorm || norm = 0.0 then Array.copy v
        else v |> Array.map (fun x -> x * maxNorm / norm)

    let softmax (m: float[,]) =
        createMatrix (rows m) (cols m) (fun r c ->
            let mutable maxV = m[r, 0]
            for j in 1 .. cols m - 1 do
                if m[r, j] > maxV then maxV <- m[r, j]

            let mutable denom = 0.0
            for j in 0 .. cols m - 1 do
                denom <- denom + exp (m[r, j] - maxV)

            exp (m[r, c] - maxV) / denom)

    let argmaxRow (m: float[,]) row =
        let mutable idx = 0
        let mutable best = m[row, 0]
        for c in 1 .. cols m - 1 do
            if m[row, c] > best then
                best <- m[row, c]
                idx <- c
        idx
