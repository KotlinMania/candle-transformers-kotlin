#if canImport(Testing)
import Testing
import CandleTransformers

@Suite("CandleTransformers Swift Export Tests")
struct CandleTransformersExportTests {
    @Test("CandleTransformers swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        let kp = KeyPoint(x: 1.0, y: 2.0, mask: 1.0)
        #expect(kp.x == 1.0)
        let cfg = models.T5Config.Companion.shared.musicgenSmall()
        #expect(cfg.dModel == 768)
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleTransformers

final class CandleTransformersExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        let kp = KeyPoint(x: 1.0, y: 2.0, mask: 1.0)
        XCTAssertEqual(kp.x, 1.0)
        let cfg = T5Config.Companion.shared.musicgenSmall()
        XCTAssertEqual(cfg.dModel, 768)
    }
}
#endif
