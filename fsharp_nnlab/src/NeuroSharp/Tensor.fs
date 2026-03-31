namespace NeuroSharp

type Batch =
    {
        X: float[,]
        Y: float[,]
    }

module Tensor =
    let fromJagged (data: float[][]) =
        let rows = data.Length
        let cols = data[0].Length
        Array2D.init rows cols (fun r c -> data[r][c])

    let toJagged (m: float[,]) =
        Array.init (Array2D.length1 m) (fun r ->
            Array.init (Array2D.length2 m) (fun c -> m[r, c]))

    let ofLabels (classCount: int) (labels: int[]) =
        Array2D.init labels.Length classCount (fun r c -> if labels[r] = c then 1.0 else 0.0)
