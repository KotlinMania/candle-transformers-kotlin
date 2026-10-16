#if canImport(Testing)
import Testing
import CandleTransformers

@Suite("CandleTransformers Swift Export Tests")
struct CandleTransformersExportTests {
    @Test("CandleTransformers swift module imported cleanly")
    func testSwiftModuleLoads() throws {
        #expect(Bool(true), "CandleTransformers swift module imported cleanly")
    }
}
#elseif canImport(XCTest)
import XCTest
import CandleTransformers

final class CandleTransformersExportTests: XCTestCase {
    func testSwiftModuleLoads() throws {
        XCTAssertTrue(true, "CandleTransformers swift module imported cleanly")
    }
}
#endif
