import Foundation
@preconcurrency import CoreML

struct Case: Decodable { let id: String; let texts: [String]; let ids: [[Int32]] }
struct Fixture: Decodable { let cases: [Case] }
public enum EmbeddingGemma2Error: Error { case predictionFailed(String) }
public enum EmbeddingGemma2Prompt: Sendable {
    case none
    func apply(to text: String) -> String { text }
}
public final class EmbeddingGemma2Tokenizer: Sendable {
    let lookup: [String: [Int32]]
    let padId: Int32 = 0
    init(_ lookup: [String: [Int32]]) { self.lookup = lookup }
    func encode(_ text: String, maxLength: Int) -> [Int32] { lookup[text] ?? [] }
}

func progress(_ text: String) { FileHandle.standardError.write(Data((text + "\n").utf8)) }

func prepare(_ root: URL) throws -> (Fixture, EmbeddingGemma2Tokenizer) {
    let fixture = try JSONDecoder().decode(Fixture.self, from: Data(contentsOf: root.appendingPathComponent("fixtures.json")))
    var lookup: [String: [Int32]] = [:]
    for c in fixture.cases {
        guard c.texts.count == c.ids.count && !c.texts.isEmpty && c.texts.count <= 32 else {
            throw EmbeddingGemma2Error.predictionFailed("invalid fixture")
        }
        for (text, ids) in zip(c.texts, c.ids) {
            guard !ids.isEmpty && ids.count <= 512 && ids.allSatisfy({ $0 >= 0 && $0 < 262144 }) else {
                throw EmbeddingGemma2Error.predictionFailed("token bound")
            }
            lookup[text] = ids
        }
    }
    return (fixture, EmbeddingGemma2Tokenizer(lookup))
}

func preferredDevices(_ plan: MLComputePlan) -> [String: Int] {
    var counts: [String: Int] = [:]
    func walk(_ block: MLModelStructure.Program.Block) {
        for op in block.operations {
            if let usage = plan.deviceUsage(for: op) { counts[usage.preferred.description, default: 0] += 1 }
            for child in op.blocks { walk(child) }
        }
    }
    if case let .program(program) = plan.modelStructure {
        for (_, f) in program.functions { walk(f.block) }
    }
    return counts
}

func loadFunction(_ name: String, _ url: URL) async throws -> (MLModel, [String: Int]) {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    cfg.functionName = name
    progress("loading \(name)")
    let model = try await MLModel.load(contentsOf: url, configuration: cfg)
    progress("loaded \(name)")
    guard CommandLine.arguments.contains("--plans"), ["embed_32", "embed_512", "pack_256"].contains(name) else {
        return (model, [:])
    }
    let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
    return (model, preferredDevices(plan))
}

func benchmarkCase(_ c: Case, _ manager: EmbeddingGemma2Manager) async throws -> [String: Any] {
    for _ in 0..<3 { _ = try await manager.embed(c.texts) }
    var ms: [Double] = []
    var vectors: [[Float]] = []
    for _ in 0..<5 {
        let started = Date()
        vectors = try await manager.embed(c.texts)
        ms.append(Date().timeIntervalSince(started) * 1000)
    }
    guard vectors.count == c.texts.count && vectors.allSatisfy({ $0.count == 768 && $0.allSatisfy(\.isFinite) }) else {
        throw EmbeddingGemma2Error.predictionFailed("bad vector")
    }
    progress("\(c.id): \(ms)")
    return ["id": c.id, "ms": ms, "vectors": vectors]
}

func benchmarkArm(_ fixture: Fixture, _ manager: EmbeddingGemma2Manager, _ packing: Bool) async throws -> [String: Any] {
    var results: [[String: Any]] = []
    for c in fixture.cases { results.append(try await benchmarkCase(c, manager)) }
    return ["runtime": packing ? "coreml-fp16-packed-ane-requested" : "coreml-fp16-single-ane-requested", "cases": results]
}

@main struct Main {
    static func main() async throws {
        let root = URL(fileURLWithPath: ProcessInfo.processInfo.environment["EG2_BENCH_WORK_DIR"] ?? "/tmp/vqtrs-apple-bench", isDirectory: true)
        let (fixture, tokenizer) = try prepare(root)
        let start = Date()
        let compiled = try await MLModel.compileModel(at: root.appendingPathComponent("coreml/EmbeddingGemma2Text.mlpackage"))
        let compileMS = Date().timeIntervalSince(start) * 1000
        progress("compiled \(compileMS) ms")
        let table = try Data(contentsOf: root.appendingPathComponent("coreml/embeddings.bf16"), options: .alwaysMapped)
        guard table.count == 262144 * 512 * 2 else { throw EmbeddingGemma2Error.predictionFailed("table shape") }
        let loadStart = Date()
        var models: [Int: MLModel] = [:]
        var placement: [String: [String: Int]] = [:]
        for n in EmbeddingGemma2Manager.lengths {
            let name = "embed_\(n)"
            let (model, counts) = try await loadFunction(name, compiled)
            models[n] = model
            placement[name] = counts
        }
        let (packed, counts) = try await loadFunction("pack_256", compiled)
        placement["pack_256"] = counts
        let loadMS = Date().timeIntervalSince(loadStart) * 1000
        var arms: [[String: Any]] = []
        for packing in [false, true] {
            let manager = EmbeddingGemma2Manager(tokenizer: tokenizer, models: models, packModel: packing ? packed : nil, table: table)
            arms.append(try await benchmarkArm(fixture, manager, packing))
        }
        let out: [String: Any] = ["compile_ms": compileMS, "load_ms": loadMS, "dimensions": 768,
            "compute_plan_preferred_devices": placement, "warmups": 3, "trials": 5, "max_inflight": 1,
            "inference_cache": false, "timing": "pretokenized IDs -> table gather + CoreML prediction + synchronized host copy; excludes tokenization and transport", "arms": arms]
        try JSONSerialization.data(withJSONObject: out, options: [.prettyPrinted, .sortedKeys]).write(to: root.appendingPathComponent("coreml.json"))
    }
}
