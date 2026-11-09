// Records every GPU contact between convex cores and triangle meshes, for byte comparison
// between PhysXGpu libraries that must produce identical contacts.
//
// Kinematic convex cores of all six core types, several sizes, margins and orientations
// (including near-axial tilts) are lowered towards three meshes in small steps, so each pair
// crosses the contact distance. Dynamic cores are dropped onto the same meshes as well.
// The meshes cover a mesh near the origin, a mesh at large coordinates and a mesh with a
// non-uniform, rotated scale. All contact points of every step are written in a sorted order.
#include "PxPhysicsAPI.h"
#include "cooking/PxCooking.h"
#undef NDEBUG
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
using namespace physx;

struct Errors : PxErrorCallback
{
	unsigned failures = 0;
	void reportError(PxErrorCode::Enum code, const char* text, const char*, int) override
	{
		std::fprintf(stderr, "%s\n", text);
		if (code != PxErrorCode::eDEBUG_INFO && code != PxErrorCode::eDEBUG_WARNING)
			++failures;
	}
};

struct Record
{
	unsigned step, actor, faceIndex, pad;
	float position[3], separation, normal[3];
	bool operator<(const Record& o) const { return std::memcmp(this, &o, sizeof(Record)) < 0; }
};

struct Contacts : PxSimulationEventCallback
{
	std::vector<Record> records;
	unsigned step = 0;
	void onContact(const PxContactPairHeader& header, const PxContactPair* pairs, PxU32 count) override
	{
		for (PxU32 i = 0; i < count; ++i)
		{
			PxContactPairPoint points[64];
			const PxU32 n = pairs[i].extractContacts(points, 64);
			assert(n <= 64);
			const bool firstIsCore = header.actors[0]->getType() != PxActorType::eRIGID_STATIC;
			const PxRigidActor* core = static_cast<const PxRigidActor*>(header.actors[firstIsCore ? 0 : 1]);
			const unsigned id = unsigned(reinterpret_cast<size_t>(core->userData));
			for (PxU32 j = 0; j < n; ++j)
			{
				Record r = {};
				r.step = step;
				r.actor = id;
				r.faceIndex = firstIsCore ? points[j].internalFaceIndex1 : points[j].internalFaceIndex0;
				std::memcpy(r.position, &points[j].position.x, sizeof(r.position));
				r.separation = points[j].separation;
				std::memcpy(r.normal, &points[j].normal.x, sizeof(r.normal));
				records.push_back(r);
			}
		}
	}
	void onConstraintBreak(PxConstraintInfo*, PxU32) override {}
	void onWake(PxActor**, PxU32) override {}
	void onSleep(PxActor**, PxU32) override {}
	void onTrigger(PxTriggerPair*, PxU32) override {}
	void onAdvance(const PxRigidBody* const*, const PxTransform*, const PxU32) override {}
};

static PxFilterFlags filterShader(PxFilterObjectAttributes a0, PxFilterData, PxFilterObjectAttributes a1,
	PxFilterData, PxPairFlags& flags, const void*, PxU32)
{
	const bool kinematic = PxFilterObjectIsKinematic(a0) || PxFilterObjectIsKinematic(a1);
	flags = (kinematic ? PxPairFlags(PxPairFlag::eDETECT_DISCRETE_CONTACT) : PxPairFlags(PxPairFlag::eCONTACT_DEFAULT))
		| PxPairFlag::eNOTIFY_TOUCH_FOUND | PxPairFlag::eNOTIFY_TOUCH_PERSISTS | PxPairFlag::eNOTIFY_CONTACT_POINTS;
	return PxFilterFlag::eDEFAULT;
}

static unsigned gSeed = 12345u;
static float uniform()
{
	gSeed = gSeed * 1664525u + 1013904223u;
	return float(gSeed >> 8) / float(1u << 24);
}

static const int GRID = 48;
static const float SPACING = 0.25f;

static float heightAt(float x, float z)
{
	return 0.08f * std::sin(1.3f * x) * std::cos(0.9f * z) + 0.03f * std::sin(4.1f * x + 2.3f * z);
}

static PxTriangleMesh* createMesh(PxPhysics& physics)
{
	std::vector<PxVec3> vertices;
	std::vector<PxU32> indices;
	for (int i = 0; i <= GRID; ++i)
		for (int j = 0; j <= GRID; ++j)
		{
			const float x = (i - GRID / 2) * SPACING, z = (j - GRID / 2) * SPACING;
			vertices.push_back(PxVec3(x, heightAt(x, z) + 0.004f * (uniform() - 0.5f), z));
		}
	for (int i = 0; i < GRID; ++i)
		for (int j = 0; j < GRID; ++j)
		{
			const PxU32 a = i * (GRID + 1) + j, b = a + 1, c = a + GRID + 1, d = c + 1;
			const PxU32 tris[6] = { a, b, c, b, d, c };
			indices.insert(indices.end(), tris, tris + 6);
		}
	PxTriangleMeshDesc desc;
	desc.points.count = PxU32(vertices.size());
	desc.points.stride = sizeof(PxVec3);
	desc.points.data = vertices.data();
	desc.triangles.count = PxU32(indices.size() / 3);
	desc.triangles.stride = 3 * sizeof(PxU32);
	desc.triangles.data = indices.data();
	PxCookingParams params(physics.getTolerancesScale());
	params.buildGPUData = true;
	PxTriangleMesh* mesh = PxCreateTriangleMesh(params, desc, physics.getPhysicsInsertionCallback());
	assert(mesh);
	return mesh;
}

static PxConvexCoreGeometry makeCore(unsigned type, float size, float margin)
{
	switch (type)
	{
		// A point or segment core needs a margin to have volume.
		case 0: return PxConvexCoreGeometry(PxConvexCore::Point(), PxMax(margin, 0.02f * size));
		case 1: return PxConvexCoreGeometry(PxConvexCore::Segment(size), PxMax(margin, 0.01f * size));
		case 2: return PxConvexCoreGeometry(PxConvexCore::Box(size, 0.4f * size, 0.7f * size), margin);
		case 3: return PxConvexCoreGeometry(PxConvexCore::Ellipsoid(0.5f * size, 0.2f * size, 0.35f * size), margin);
		case 4: return PxConvexCoreGeometry(PxConvexCore::Cylinder(0.05f * size, 0.5f * size), margin);
		default: return PxConvexCoreGeometry(PxConvexCore::Cone(0.6f * size, 0.3f * size), margin);
	}
}

// Targeted scene for the cylinder/cone support approximation: a flat core with a tiny tilt moves
// along its axis by 1e-7 m per step between two walls, so each wall crosses the contact distance
// within the narrow band where sampled bounds lie inside the core. Both walls share a BVH leaf, so
// the leaf is always traversed and each wall triangle reaches the per-triangle test.
static int runBoundaryScene(PxPhysics& physics, PxScene& scene, PxMaterial& material, Contacts& contacts,
	const char* path, Errors& errors)
{
	const float wall = 0.005f + 0.002f + 3e-5f;	// half height + margin + contact distance + start gap
	const float travel = 6e-5f;
	const PxVec3 vertices[8] = {
		PxVec3(wall, -1.5f, -1.5f), PxVec3(wall, 1.5f, -1.5f), PxVec3(wall, 1.5f, 1.5f), PxVec3(wall, -1.5f, 1.5f),
		PxVec3(-wall + travel, -1.5f, -1.5f), PxVec3(-wall + travel, 1.5f, -1.5f),
		PxVec3(-wall + travel, 1.5f, 1.5f), PxVec3(-wall + travel, -1.5f, 1.5f) };
	// Both walls face the core: the +X wall towards -X, the -X wall towards +X.
	const PxU32 indices[12] = { 0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7 };
	PxTriangleMeshDesc desc;
	desc.points.count = 8;
	desc.points.stride = sizeof(PxVec3);
	desc.points.data = vertices;
	desc.triangles.count = 4;
	desc.triangles.stride = 3 * sizeof(PxU32);
	desc.triangles.data = indices;
	PxCookingParams params(physics.getTolerancesScale());
	params.buildGPUData = true;
	PxTriangleMesh* mesh = PxCreateTriangleMesh(params, desc, physics.getPhysicsInsertionCallback());
	assert(mesh);

	struct Case { PxConvexCoreGeometry geometry; PxQuat rotation; };
	const PxConvexCoreGeometry cylinder(PxConvexCore::Cylinder(0.01f, 1.0f), 0.001f);
	const PxConvexCoreGeometry smallCylinder(PxConvexCore::Cylinder(0.01f, 0.25f), 0.001f);
	const PxConvexCoreGeometry cone(PxConvexCore::Cone(0.01f, 1.0f), 0.001f);
	const Case cases[] = {
		{ cylinder, PxQuat(PxIdentity) },
		{ cylinder, PxQuat(8e-6f, PxVec3(0, 0, 1)) },
		{ cylinder, PxQuat(-8e-6f, PxVec3(0, 0, 1)) },
		{ cylinder, PxQuat(8e-6f, PxVec3(0, 1, 0)) },
		{ cylinder, PxQuat(5e-6f, PxVec3(0, 0, 1)) },
		{ smallCylinder, PxQuat(8e-6f, PxVec3(0, 0, 1)) },
		{ cone, PxQuat(PxIdentity) },
		{ cone, PxQuat(8e-6f, PxVec3(0, 0, 1)) },
		{ cone, PxQuat(-8e-6f, PxVec3(0, 0, 1)) } };
	const unsigned count = sizeof(cases) / sizeof(cases[0]);
	std::vector<PxRigidDynamic*> bodies;
	std::vector<PxTransform> starts;
	for (unsigned k = 0; k < count; ++k)
	{
		const PxVec3 origin(0, 0, 5.0f * float(k));
		PxRigidStatic* walls = physics.createRigidStatic(PxTransform(origin));
		PxShape* wallShape = PxRigidActorExt::createExclusiveShape(*walls, PxTriangleMeshGeometry(mesh), material);
		assert(wallShape);
		wallShape->setContactOffset(0.0005f);
		wallShape->setRestOffset(0.0f);
		scene.addActor(*walls);
		const PxTransform start(origin, cases[k].rotation);
		PxRigidDynamic* body = physics.createRigidDynamic(start);
		body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
		PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, cases[k].geometry, material);
		assert(shape);
		shape->setContactOffset(0.0005f);
		shape->setRestOffset(0.0f);
		body->userData = reinterpret_cast<void*>(size_t(k));
		scene.addActor(*body);
		bodies.push_back(body);
		starts.push_back(start);
	}
	const unsigned steps = 600;
	for (unsigned step = 0; step < steps; ++step)
	{
		contacts.step = step;
		for (unsigned k = 0; k < count; ++k)
			bodies[k]->setKinematicTarget(PxTransform(starts[k].p + PxVec3(travel * float(step + 1) / float(steps), 0, 0), starts[k].q));
		scene.simulate(1.0f / 120.0f);
		assert(scene.fetchResults(true));
	}
	std::vector<Record> records = contacts.records;
	std::sort(records.begin(), records.end());
	FILE* output = std::fopen(path, "wb");
	assert(output);
	if (!records.empty())
		assert(std::fwrite(records.data(), sizeof(Record), records.size(), output) == records.size());
	std::fclose(output);
	// Steps in contact with each wall, per case. Cooking may reorder triangles, so the wall is
	// identified by the side of the contact point (all cores start at x = 0).
	for (unsigned k = 0; k < count; ++k)
	{
		std::vector<unsigned> plus, minus;
		for (const Record& r : records)
			if (r.actor == k)
				(r.position[0] > 0.0f ? plus : minus).push_back(r.step);
		for (std::vector<unsigned>* v : { &plus, &minus })
		{
			std::sort(v->begin(), v->end());
			v->erase(std::unique(v->begin(), v->end()), v->end());
		}
		std::printf("case %u steps_touching_plus_wall %zu steps_touching_minus_wall %zu\n", k, plus.size(), minus.size());
	}
	std::printf("contacts %zu\n", records.size());
	scene.release();
	mesh->release();
	assert(errors.failures == 0);
	return 0;
}

int main(int argc, char** argv)
{
	assert(argc >= 2 && "output.bin");
	PxDefaultAllocator allocator;
	Errors errors;
	PxFoundation* foundation = PxCreateFoundation(PX_PHYSICS_VERSION, allocator, errors);
	PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale());
	PxCudaContextManagerDesc cudaDesc;
	PxCudaContextManager* cuda = PxCreateCudaContextManager(*foundation, cudaDesc);
	assert(cuda && cuda->contextIsValid());
	PxDefaultCpuDispatcher* dispatcher = PxDefaultCpuDispatcherCreate(4);
	Contacts contacts;
	PxSceneDesc desc(physics->getTolerancesScale());
	desc.gravity = PxVec3(0, -9.81f, 0);
	desc.cpuDispatcher = dispatcher;
	desc.filterShader = filterShader;
	desc.simulationEventCallback = &contacts;
	desc.cudaContextManager = cuda;
	desc.flags |= PxSceneFlag::eENABLE_GPU_DYNAMICS;
	desc.broadPhaseType = PxBroadPhaseType::eGPU;
	desc.solverType = PxSolverType::eTGS;
	desc.staticKineFilteringMode = PxPairFilteringMode::eKEEP;
	PxScene* scene = physics->createScene(desc);
	assert(scene);
	PxMaterial* material = physics->createMaterial(0.6f, 0.6f, 0.0f);
	if (argc > 2 && std::strcmp(argv[2], "boundary") == 0)
		return runBoundaryScene(*physics, *scene, *material, contacts, argv[1], errors);
	PxTriangleMesh* mesh = createMesh(*physics);

	// Mesh 0 near the origin, mesh 1 at large coordinates, mesh 2 with a rotated non-uniform scale.
	const PxVec3 meshOrigins[3] = { PxVec3(0), PxVec3(2000.0f, 0, -3000.0f), PxVec3(0, 0, 40.0f) };
	const PxMeshScale meshScales[3] = { PxMeshScale(), PxMeshScale(),
		PxMeshScale(PxVec3(1.5f, 0.7f, 1.2f), PxQuat(0.3f, PxVec3(0, 1, 0))) };
	for (int m = 0; m < 3; ++m)
	{
		PxRigidStatic* ground = physics->createRigidStatic(PxTransform(meshOrigins[m]));
		PxShape* shape = PxRigidActorExt::createExclusiveShape(*ground, PxTriangleMeshGeometry(mesh, meshScales[m]), *material);
		shape->setContactOffset(0.02f);
		shape->setRestOffset(0.0f);
		scene->addActor(*ground);
	}

	// Orientations: identity, near-axial tilts that exercise the approximate support branches,
	// a core axis turned vertical with a tilt, and a generic rotation.
	const PxQuat orientations[5] = {
		PxQuat(PxIdentity),
		PxQuat(8e-6f, PxVec3(0, 0, 1)),
		PxQuat(PxHalfPi, PxVec3(0, 0, 1)) * PxQuat(-5e-6f, PxVec3(0, 1, 0)),
		PxQuat(PxHalfPi + 7e-6f, PxVec3(0, 0, 1)),
		PxQuat(0.7f, PxVec3(0.3f, 0.8f, 0.5f).getNormalized()) };
	const float sizes[3] = { 0.06f, 0.5f, 2.5f };
	const float margins[2] = { 0.0f, 0.01f };
	struct Lowered { PxRigidDynamic* actor; PxTransform start; };
	std::vector<Lowered> lowered;
	std::vector<PxRigidDynamic*> dropped;
	unsigned id = 0;
	for (int m = 0; m < 3; ++m)
		for (unsigned type = 0; type < 6; ++type)
			for (float size : sizes)
				for (float margin : margins)
					for (const PxQuat& q : orientations)
					{
						const float x = (uniform() - 0.5f) * 8.0f, z = (uniform() - 0.5f) * 8.0f;
						const PxVec3 ground = meshOrigins[m] + meshScales[m].transform(PxVec3(x, heightAt(x, z), z));
						PxTransform pose(ground, q.getNormalized());
						PxRigidDynamic* body = physics->createRigidDynamic(pose);
						body->setRigidBodyFlag(PxRigidBodyFlag::eKINEMATIC, true);
						PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, makeCore(type, size, margin), *material);
						assert(shape);
						shape->setContactOffset(0.02f);
						shape->setRestOffset(0.0f);
						// Start with the bottom of the bounds 3 cm above the mesh below the center.
						const PxBounds3 bounds = PxShapeExt::getWorldBounds(*shape, *body, 1.0f);
						pose.p.y += ground.y + 0.03f - bounds.minimum.y;
						body->setGlobalPose(pose);
						body->userData = reinterpret_cast<void*>(size_t(id++));
						scene->addActor(*body);
						lowered.push_back({ body, pose });
					}
	for (int m = 0; m < 3; ++m)
		for (unsigned type = 0; type < 6; ++type)
		{
			const float x = (uniform() - 0.5f) * 8.0f, z = (uniform() - 0.5f) * 8.0f;
			const PxTransform pose(meshOrigins[m] + PxVec3(x, 1.0f, z),
				PxQuat(uniform() * PxTwoPi, PxVec3(uniform(), uniform(), uniform() + 0.1f).getNormalized()));
			PxRigidDynamic* body = physics->createRigidDynamic(pose);
			PxShape* shape = PxRigidActorExt::createExclusiveShape(*body, makeCore(type, 0.3f, 0.005f), *material);
			assert(shape);
			shape->setContactOffset(0.02f);
			shape->setRestOffset(0.0f);
			PxRigidBodyExt::updateMassAndInertia(*body, 500.0f);
			body->setSleepThreshold(0);
			body->userData = reinterpret_cast<void*>(size_t(id++));
			scene->addActor(*body);
			dropped.push_back(body);
		}

	const unsigned steps = 240;
	for (unsigned step = 0; step < steps; ++step)
	{
		contacts.step = step;
		// Lower by 0.25 mm per step: every kinematic pair crosses the contact distance slowly.
		for (const Lowered& l : lowered)
			l.actor->setKinematicTarget(PxTransform(l.start.p - PxVec3(0, 0.00025f * float(step + 1), 0), l.start.q));
		scene->simulate(1.0f / 120.0f);
		assert(scene->fetchResults(true));
	}
	std::vector<Record> records = contacts.records;
	std::sort(records.begin(), records.end());
	FILE* output = std::fopen(argv[1], "wb");
	assert(output);
	if (!records.empty())
		assert(std::fwrite(records.data(), sizeof(Record), records.size(), output) == records.size());
	for (PxRigidDynamic* body : dropped)
	{
		const PxTransform pose = body->getGlobalPose();
		assert(std::fwrite(&pose, sizeof(pose), 1, output) == 1);
	}
	std::fclose(output);
	unsigned perType[6] = {};
	for (const Record& r : records)
		if (r.actor < lowered.size())
			++perType[(r.actor / (3 * 2 * 5)) % 6];
	std::printf("contacts %zu kinematic_by_type %u %u %u %u %u %u actors %u\n", records.size(), perType[0], perType[1],
		perType[2], perType[3], perType[4], perType[5], id);
	scene->release();
	mesh->release();
	material->release();
	physics->release();
	dispatcher->release();
	cuda->release();
	foundation->release();
	assert(errors.failures == 0);
	return 0;
}
