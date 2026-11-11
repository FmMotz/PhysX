// Differential test for Gu::ConvexCore::TriangleCore against the identity PointsCore
// it replaces in the GPU convex core vs triangle mesh kernels.
//
// Build against a checkout of the gpu-convexcore-trimesh-triangle-core branch:
//   g++ -std=c++14 -O2 -I$SDK/include -I$SDK/source/geomutils/include
//       -I$SDK/source/common/include -I$SDK/source/common/src
//       triangle_core_support_test.cpp -o triangle_core_support_test
//   ./triangle_core_support_test [random-case-count] [--strict]
//
// For every direction and triangle, localSupport and contactFace must return the same values
// for both cores. The PointsCore mirrors the stock GPU kernel: three float4 vertices (16 byte
// stride), unit scale and identity rotation. Bit patterns are compared. Contact faces must match
// exactly. For support points the only accepted difference is the sign of zero, because the
// identity quaternion rotation in PointsCore turns -0.0 into +0.0 (vx * 0.5 + (0 * vz - 0 * vy)
// * 1 + 0 * dot2) while TriangleCore returns the stored vertex. Such differences are counted,
// must come from a triangle with a -0.0 vertex component, and --strict makes them fail the run.
// Coordinates stay below FLT_MAX / 2 because the identity rotation doubles them first. The kernels
// are built with -ftz=true, which flushes subnormal coordinates in the PointsCore path on the
// device; this host build does not reproduce that. This test is compiled for the host with g++; the kernels are compiled by nvcc and
// their device behaviour is covered by the byte-identical GPU contact outputs, not by this test.

#include "GuConvexSupport.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

using namespace physx;
using namespace physx::Gu;

namespace
{
	struct Float4 { float x, y, z, w; };

	struct Failure { const char* what; PxU32 caseIndex; };

	bool sameBits(const PxVec3& a, const PxVec3& b)
	{
		return memcmp(&a, &b, sizeof(PxVec3)) == 0;
	}

	// Equal values where every bit difference is the sign of a zero component.
	bool sameUpToSignedZero(const PxVec3& a, const PxVec3& b)
	{
		const float* pa = &a.x; const float* pb = &b.x;
		for (int i = 0; i < 3; ++i)
			if (memcmp(pa + i, pb + i, sizeof(float)) != 0 && !(pa[i] == 0.0f && pb[i] == 0.0f))
				return false;
		return true;
	}

	struct Case
	{
		const char* name;
		PxVec3 v[3];
		PxVec3 dir;
	};

	struct Result
	{
		PxVec3 support;
		PxU32 faceCount;
		PxVec3 faceNormal;
		PxVec3 facePoints[ConvexCore::MAX_FACE_POINTS];
	};

	struct Stats { PxU32 signedZeroSupport, signedZeroWithoutNegativeZeroVertex; };

	bool hasNegativeZeroComponent(const Case& c)
	{
		for (PxU32 i = 0; i < 3; ++i)
		{
			const float* p = &c.v[i].x;
			for (int k = 0; k < 3; ++k)
				if (p[k] == 0.0f && std::signbit(p[k]))
					return true;
		}
		return false;
	}

	Result evalTriangle(const Case& c)
	{
		const ConvexCore::TriangleCore core = { { c.v[0].x, c.v[0].y, c.v[0].z,
			c.v[1].x, c.v[1].y, c.v[1].z, c.v[2].x, c.v[2].y, c.v[2].z } };
		PxU8 data[ConvexCore::MAX_CORE_SIZE]; memset(data, 0xcd, sizeof(data));
		memcpy(data, &core, sizeof(core));
		Result r; memset(static_cast<void*>(&r), 0xee, sizeof(r));
		r.support = ConvexCore::localSupport(ConvexCore::Type::eTRIANGLE)(c.dir, data);
		r.faceCount = ConvexCore::contactFace(ConvexCore::Type::eTRIANGLE)(c.dir, r.support, data,
			r.faceNormal, r.facePoints);
		return r;
	}

	Result evalPoints(const Case& c)
	{
		Float4 verts[3];
		for (PxU32 i = 0; i < 3; ++i)
		{
			verts[i].x = c.v[i].x; verts[i].y = c.v[i].y; verts[i].z = c.v[i].z; verts[i].w = 1.0f;
		}
		ConvexCore::PointsCore core;
		core.points = verts;
		core.numPoints = 3;
		core.stride = sizeof(Float4);
		core.S = PxVec3(1);
		core.R = PxQuat(PxIdentity);
		core.pad[0] = core.pad[1] = 0;
		PxU8 data[ConvexCore::MAX_CORE_SIZE]; memset(data, 0xcd, sizeof(data));
		memcpy(data, &core, sizeof(core));
		Result r; memset(static_cast<void*>(&r), 0xee, sizeof(r));
		r.support = ConvexCore::localSupport(ConvexCore::Type::ePOINTS)(c.dir, data);
		r.faceCount = ConvexCore::contactFace(ConvexCore::Type::ePOINTS)(c.dir, r.support, data,
			r.faceNormal, r.facePoints);
		return r;
	}

	bool compare(const Case& c, PxU32 caseIndex, std::vector<Failure>& failures, Stats& stats)
	{
		const Result t = evalTriangle(c);
		const Result p = evalPoints(c);
		bool ok = true;
		if (!sameBits(t.support, p.support))
		{
			if (sameUpToSignedZero(t.support, p.support))
			{
				if (stats.signedZeroSupport < 5 && getenv("SHOW_SIGNED_ZERO"))
					printf("signed zero %s: v0 (%g %g %g) v1 (%g %g %g) v2 (%g %g %g) dir (%g %g %g) tri (%g %g %g) pts (%g %g %g)\n",
						c.name, c.v[0].x, c.v[0].y, c.v[0].z, c.v[1].x, c.v[1].y, c.v[1].z, c.v[2].x, c.v[2].y, c.v[2].z,
						c.dir.x, c.dir.y, c.dir.z, t.support.x, t.support.y, t.support.z, p.support.x, p.support.y, p.support.z);
				++stats.signedZeroSupport;
				// The only known mechanism is a stored -0.0 vertex component; anything else is a failure.
				if (!hasNegativeZeroComponent(c))
				{
					++stats.signedZeroWithoutNegativeZeroVertex;
					failures.push_back({ "signed zero without -0.0 vertex", caseIndex }); ok = false;
				}
			}
			else
			{
				failures.push_back({ "support", caseIndex }); ok = false;
			}
		}
		if (t.faceCount != p.faceCount)
		{
			failures.push_back({ "face count", caseIndex }); ok = false;
		}
		else if (t.faceCount > 0)
		{
			// Both cores compute the face through the same PointsCore code, so faces must match bit for bit.
			if (!sameBits(t.faceNormal, p.faceNormal))
			{
				failures.push_back({ "face normal", caseIndex }); ok = false;
			}
			for (PxU32 i = 0; i < t.faceCount; ++i)
				if (!sameBits(t.facePoints[i], p.facePoints[i]))
				{
					failures.push_back({ "face point", caseIndex }); ok = false; break;
				}
		}
		if (!ok)
			printf("FAIL %s: dir (%g %g %g) support tri (%g %g %g) pts (%g %g %g) faces %u/%u\n",
				c.name, c.dir.x, c.dir.y, c.dir.z, t.support.x, t.support.y, t.support.z,
				p.support.x, p.support.y, p.support.z, t.faceCount, p.faceCount);
		return ok;
	}

	void addFixedCases(std::vector<Case>& cases)
	{
		const PxVec3 a(0, 0, 0), b(1, 0, 0), c(0, 1, 0);
		const PxVec3 big(3.0e8f, -2.0e8f, 1.0e8f);
		const PxVec3 tiny(1.0e-30f, -1.0e-30f, 1.0e-30f);
		const float eps = FLT_EPSILON;
		const Case fixed[] = {
			// Unique maxima along each vertex
			{ "unique v0", { a, b, c }, PxVec3(-1, -1, 0) },
			{ "unique v1", { a, b, c }, PxVec3(1, 0, 0) },
			{ "unique v2", { a, b, c }, PxVec3(0, 1, 0) },
			// Exact ties: the first maximum in index order must win in both cores
			{ "tie v0 v1", { a, b, c }, PxVec3(0, -1, 0) },
			{ "tie v1 v2", { a, b, c }, PxVec3(1, 1, 0) },
			{ "tie v0 v2", { a, b, c }, PxVec3(-1, 0, 0) },
			{ "tie all (face normal)", { a, b, c }, PxVec3(0, 0, 1) },
			{ "tie all (negative face normal)", { a, b, c }, PxVec3(0, 0, -1) },
			// Face queries from directions near and off the face normal
			{ "near normal", { a, b, c }, PxVec3(0.01f, 0.02f, 1) },
			{ "edge normal", { a, b, c }, PxVec3(1, 1, 0.05f) },
			{ "grazing", { a, b, c }, PxVec3(1, 0, 0.001f) },
			// Zero and tiny directions, including just below and above the FLT_EPSILON guard
			{ "zero dir", { a, b, c }, PxVec3(0) },
			{ "tiny dir", { a, b, c }, PxVec3(0, 0, eps * 0.5f) },
			{ "near eps dir below", { a, b, c }, PxVec3(0, 0, nextafterf(eps, 0.0f)) },
			{ "near eps dir above", { a, b, c }, PxVec3(0, 0, nextafterf(eps, 1.0f)) },
			{ "tiny diagonal", { a, b, c }, PxVec3(eps * 0.5f, eps * 0.5f, eps * 0.5f) },
			{ "denormal dir", { a, b, c }, PxVec3(1.0e-40f, 0, 0) },
			// Epsilon boundary on a triangle away from the origin: below returns (0,0,0), at and above a vertex
			{ "eps below, offset triangle", { b + c, b + b, c + c }, PxVec3(nextafterf(eps, 0.0f), 0, 0) },
			{ "eps exact, offset triangle", { b + c, b + b, c + c }, PxVec3(eps, 0, 0) },
			{ "eps above, offset triangle", { b + c, b + b, c + c }, PxVec3(nextafterf(eps, 1.0f), 0, 0) },
			// Repeated and collinear vertices
			{ "repeated v0 v1", { a, a, c }, PxVec3(1, 1, 0) },
			{ "repeated v1 v2", { a, b, b }, PxVec3(1, 0, 0) },
			{ "all repeated", { b, b, b }, PxVec3(0, 0, 1) },
			{ "all repeated zero", { a, a, a }, PxVec3(0, 0, 1) },
			{ "collinear", { a, b, PxVec3(2, 0, 0) }, PxVec3(1, 0, 0) },
			{ "collinear sideways", { a, b, PxVec3(2, 0, 0) }, PxVec3(0, 1, 0) },
			{ "collinear reversed", { PxVec3(2, 0, 0), b, a }, PxVec3(-1, 0, 0) },
			// Large and tiny coordinates
			{ "large coords", { big, -big, PxVec3(0, 0, 3.0e8f) }, PxVec3(0.3f, -0.6f, 0.2f) },
			{ "large coords face", { big, big + b, big + c }, PxVec3(0, 0, 1) },
			{ "tiny coords", { tiny, -tiny, PxVec3(0) }, PxVec3(1, 0, 0) },
			{ "mixed coords", { big, tiny, PxVec3(-1, 2, -3) }, PxVec3(-1, 1, -1) },
			// Negative zero: PointsCore returns +0.0 here, TriangleCore keeps -0.0
			{ "negative zero", { PxVec3(-0.0f, 0, 0), b, c }, PxVec3(-1, 0, 0) },
			{ "negative zero all", { PxVec3(-0.0f, -0.0f, -0.0f), b, c }, PxVec3(-1, -1, -1) },
			{ "negative zero dir", { a, b, c }, PxVec3(-0.0f, 1, 0) },
		};
		for (const Case& f : fixed)
			cases.push_back(f);
	}

	void addRandomCases(std::vector<Case>& cases, PxU32 count)
	{
		std::mt19937 rng(20261111u);
		std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
		std::uniform_int_distribution<int> expo(-20, 20);
		std::uniform_int_distribution<int> kind(0, 7);
		std::uniform_int_distribution<int> zeroAxis(0, 35);
		for (PxU32 n = 0; n < count; ++n)
		{
			Case c; c.name = "random";
			const float scale = ldexpf(1.0f, expo(rng));
			for (PxU32 i = 0; i < 3; ++i)
				c.v[i] = PxVec3(unit(rng), unit(rng), unit(rng)) * scale;
			switch (kind(rng))
			{
				case 0: c.dir = PxVec3(unit(rng), unit(rng), unit(rng)); break;
				case 1: c.dir = PxVec3(unit(rng), unit(rng), unit(rng)).getNormalized(); break;
				case 2: c.dir = (c.v[1] - c.v[0]).cross(c.v[2] - c.v[0]); break;              // face normal
				case 3: c.dir = -(c.v[1] - c.v[0]).cross(c.v[2] - c.v[0]).getNormalized(); break;
				case 4: c.dir = c.v[n % 3] - c.v[(n + 1) % 3]; break;                          // along an edge
				case 5: c.v[1] = c.v[0]; c.dir = PxVec3(unit(rng), unit(rng), unit(rng)); break; // repeated
				case 6: c.v[2] = c.v[0] + (c.v[1] - c.v[0]) * 2.0f;                            // collinear
						c.dir = PxVec3(unit(rng), unit(rng), unit(rng)); break;
				default: c.dir = PxVec3(unit(rng), unit(rng), unit(rng)) * ldexpf(1.0f, -expo(rng)); break;
			}
			// A quarter of the cases get one negative zero vertex component
			const int axis = zeroAxis(rng);
			if (axis < 9)
				(&c.v[axis / 3].x)[axis % 3] = -0.0f;
			cases.push_back(c);
		}
	}
}

int main(int argc, char** argv)
{
	PxU32 randomCount = 200000u;
	bool strict = false;
	for (int i = 1; i < argc; ++i)
	{
		if (strcmp(argv[i], "--strict") == 0) strict = true;
		else randomCount = PxU32(atoi(argv[i]));
	}
	std::vector<Case> cases;
	addFixedCases(cases);
	const PxU32 fixedCount = PxU32(cases.size());
	addRandomCases(cases, randomCount);

	std::vector<Failure> failures;
	Stats stats = { 0, 0 };
	PxU32 nonZeroSupport = 0, faces[4] = { 0, 0, 0, 0 };
	for (PxU32 i = 0; i < cases.size(); ++i)
	{
		compare(cases[i], i, failures, stats);
		const Result t = evalTriangle(cases[i]);
		if (!sameBits(t.support, PxVec3(0))) ++nonZeroSupport;
		faces[t.faceCount < 3 ? t.faceCount : 3]++;
	}
	printf("cases: %u fixed + %u random, non-zero supports %u, face counts 0/1/2/3+: %u/%u/%u/%u\n",
		fixedCount, randomCount, nonZeroSupport, faces[0], faces[1], faces[2], faces[3]);
	printf("sign-of-zero only differences in support points: %u (without a -0.0 vertex: %u)\n",
		stats.signedZeroSupport, stats.signedZeroWithoutNegativeZeroVertex);
	if (strict && stats.signedZeroSupport)
	{
		printf("FAIL (strict): sign-of-zero differences present\n");
		return 1;
	}
	if (failures.empty())
	{
		printf("PASS: TriangleCore matches identity PointsCore (bit for bit except the sign of zero)\n");
		return 0;
	}
	printf("FAIL: %zu mismatches\n", failures.size());
	return 1;
}
