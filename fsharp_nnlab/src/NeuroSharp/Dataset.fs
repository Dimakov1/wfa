namespace NeuroSharp

open NeuroSharp.Math

type Dataset =
    {
        Features: float[,]
        Targets: float[,]
    }

[<RequireQualifiedAccess>]
module Dataset =
    let create x y = { Features = x; Targets = y }

    let mapFeatures f ds =
        { ds with Features = f ds.Features }

    let mapTargets f ds =
        { ds with Targets = f ds.Targets }

    let shuffle ds =
        let count = rows ds.Features
        let indices = [| 0 .. count - 1 |]
        for i in count - 1 .. -1 .. 1 do
            let j = random.Next(i + 1)
            let tmp = indices[i]
            indices[i] <- indices[j]
            indices[j] <- tmp

        let x' = createMatrix count (cols ds.Features) (fun r c -> ds.Features[indices[r], c])
        let y' = createMatrix count (cols ds.Targets) (fun r c -> ds.Targets[indices[r], c])
        create x' y'

    let split trainRatio ds =
        let n = rows ds.Features
        let nTrain = int (float n * trainRatio)
        let makePart start length =
            create
                (createMatrix length (cols ds.Features) (fun r c -> ds.Features[start + r, c]))
                (createMatrix length (cols ds.Targets) (fun r c -> ds.Targets[start + r, c]))
        makePart 0 nTrain, makePart nTrain (n - nTrain)

    let batches batchSize ds =
        seq {
            let n = rows ds.Features
            let mutable start = 0
            while start < n do
                let size = min batchSize (n - start)
                yield {
                    X = createMatrix size (cols ds.Features) (fun r c -> ds.Features[start + r, c])
                    Y = createMatrix size (cols ds.Targets) (fun r c -> ds.Targets[start + r, c])
                }
                start <- start + size
        }

    let normalizeColumns ds =
        let x = ds.Features
        let nRows = rows x
        let nCols = cols x

        let means =
            Array.init nCols (fun c ->
                let mutable acc = 0.0
                for r in 0 .. nRows - 1 do acc <- acc + x[r, c]
                acc / float nRows)

        let stds =
            Array.init nCols (fun c ->
                let mutable acc = 0.0
                for r in 0 .. nRows - 1 do
                    let d = x[r, c] - means[c]
                    acc <- acc + d * d
                let s = sqrt (acc / float nRows)
                if s < 1e-9 then 1.0 else s)

        let normalized =
            createMatrix nRows nCols (fun r c -> (x[r, c] - means[c]) / stds[c])

        { ds with Features = normalized }
