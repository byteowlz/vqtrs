import Foundation
@preconcurrency import CoreML

func deviceName(_ device: MLComputeDevice) -> String {
    switch device {
    case .cpu: return "cpu"
    case .gpu: return "gpu"
    case .neuralEngine: return "neuralEngine"
    @unknown default: return "unknown"
    }
}

func deviceCounts(_ plan: MLComputePlan) -> [String: Int] {
    guard case let .program(program) = plan.modelStructure else { return [:] }
    let operations = program.functions.values.flatMap { $0.block.operations }
    var counts: [String: Int] = [:]
    for operation in operations {
        guard let usage = plan.deviceUsage(for: operation) else { continue }
        counts[deviceName(usage.preferred), default: 0] += 1
    }
    return counts
}

@main struct Main {
    static func main() async throws {
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["EG2_BENCH_WORK_DIR"] ?? "/tmp/vqtrs-apple-bench", isDirectory: true)
        let compiled = try await MLModel.compileModel(at: root.appendingPathComponent("coreml/EmbeddingGemma2Text.mlpackage"))
        let cfg = MLModelConfiguration()
        cfg.functionName = "embed_32"
        cfg.computeUnits = .cpuAndNeuralEngine
        FileHandle.standardError.write(Data("querying embed_32 plan\n".utf8))
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: cfg)
        try JSONSerialization.data(withJSONObject: deviceCounts(plan), options: [.prettyPrinted]).write(to: root.appendingPathComponent("coreml-plan.json"))
    }
}
