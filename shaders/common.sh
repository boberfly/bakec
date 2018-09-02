// Common

uniform vec4 u_params;
#define workOffset          floatBitsToUint(u_params.x)
#define pixOffset           floatBitsToUint(u_params.x)
#define workCount           floatBitsToUint(u_params.y)
#define bvhCount            floatBitsToUint(u_params.z)

#define FLT_MAX 3.402823466e+38
#define BARY_MIN -1e-5
#define BARY_MAX 1.0

struct Pix
{
	vec3 p;
	vec3 d;
};

struct BVH
{
	float aabbMinX; float aabbMinY; float aabbMinZ;
	float aabbMaxX; float aabbMaxY; float aabbMaxZ;
	uint start;
	uint end;
	uint jump; // Index to the next BVH if we skip this subtree
};

float RayAABB(vec3 o, vec3 d, vec3 mins, vec3 maxs)
{
	//vec3 dabs = abs(d);
	vec3 t1 = (mins - o) / d;
	vec3 t2 = (maxs - o) / d;
	vec3 tmin = min(t1, t2);
	vec3 tmax = max(t1, t2);
	float a = max(tmin.x, max(tmin.y, tmin.z));
	float b = min(tmax.x, min(tmax.y, tmax.z));
	return (b >= 0 && a <= b) ? a : FLT_MAX;
}

vec3 barycentric(dvec3 p, dvec3 a, dvec3 b, dvec3 c)
{
	dvec3 v0 = b - a;
	dvec3 v1 = c - a;
	dvec3 v2 = p - a;
	double d00 = dot(v0, v0);
	double d01 = dot(v0, v1);
	double d11 = dot(v1, v1);
	double d20 = dot(v2, v0);
	double d21 = dot(v2, v1);
	double denom = d00 * d11 - d01 * d01;
	double y = (d11 * d20 - d01 * d21) / denom;
	double z = (d00 * d21 - d01 * d20) / denom;
	return vec3(dvec3(1.0 - y - z, y, z));
}

// Returns distance (x) + barycentric coordinates (yzw)
vec4 raycast(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
	vec3 n = normalize(cross(b - a, c - a));
	float nd = dot(d, n);
	if (abs(nd) > 0)
	{
		float pn = dot(o, n);
		float t = (dot(a, n) - pn) / nd;
		if (t >= 0)
		{
			vec3 p = o + d * t;
			vec3 b = barycentric(p, a, b, c);
			if (b.x >= BARY_MIN && b.y >= BARY_MIN && b.y <= BARY_MAX && b.z >= BARY_MIN && b.z <= BARY_MAX)
			{
				return vec4(t, b.x, b.y, b.z);
			}
		}
	}
	return vec4(FLT_MAX, 0, 0, 0);
}

float raycastRange(vec3 o, vec3 d, uint start, uint end, float mindist, out uint o_idx, out vec3 o_bcoord)
{
	float mint = FLT_MAX;
	for (uint tidx = start; tidx < end; tidx += 3)
	{
		vec3 v0 = positions[tidx + 0];
		vec3 v1 = positions[tidx + 1];
		vec3 v2 = positions[tidx + 2];
		vec4 r = raycast(o, d, v0, v1, v2);
		if (r.x >= mindist && r.x < mint)
		{
			mint = r.x;
			o_idx = tidx;
			o_bcoord = r.yzw;
		}
	}
	return mint;
}

float raycastBVH(vec3 o, vec3 d, float mint, in out uint o_idx, in out vec3 o_bcoord)
{
	uint i = 0;
	while (i < bvhCount)
	{
		BVH bvh = bvhs[i];
		vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
		vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
		float distAABB = RayAABB(o, d, aabbMin, aabbMax);
		if (distAABB < mint)
		//if (distAABB != FLT_MAX)
		{
			uint ridx = 0;
			vec3 rbcoord = vec3(0, 0, 0);
			float t = raycastRange(o, d, bvh.start, bvh.end, 0, ridx, rbcoord);
			if (t < mint)
			{
				mint = t;
				o_idx = ridx;
				o_bcoord = rbcoord;
			}
			++i;
		}
		else
		{
			i = bvh.jump;
		}
	}

	return mint;
}

// Returns distance only
float raycast_dist(vec3 o, vec3 d, vec3 a, vec3 b, vec3 c)
{
	vec3 n = normalize(cross(b - a, c - a));
	float nd = dot(d, n);
	if (abs(nd) > 0)
	{
		float pn = dot(o, n);
		float t = (dot(a, n) - pn) / nd;
		if (t >= 0)
		{
			vec3 p = o + d * t;
			vec3 b = barycentric(p, a, b, c);
			if (b.x >= 0 && //b.x <= 1 &&
				b.y >= 0 && b.y <= 1 &&
				b.z >= 0 && b.z <= 1)
			{
				return t;
			}
		}
	}
	return FLT_MAX;
}

float raycastRange_dist(vec3 o, vec3 d, uint start, uint end, float mindist)
{
	float mint = FLT_MAX;
	for (uint tidx = start; tidx < end; tidx += 3)
	{
		vec3 v0 = positions[tidx + 0];
		vec3 v1 = positions[tidx + 1];
		vec3 v2 = positions[tidx + 2];
		float t = raycast_dist(o, d, v0, v1, v2);
		if (t >= mindist && t < mint)
		{
			mint = t;
		}
	}
	return mint;
}

float raycastBVH_dist(vec3 o, vec3 d, float mindist, float maxdist)
{
	float mint = FLT_MAX;
	uint i = 0;
	while (i < bvhCount)
	{
		BVH bvh = bvhs[i];
		vec3 aabbMin = vec3(bvh.aabbMinX, bvh.aabbMinY, bvh.aabbMinZ);
		vec3 aabbMax = vec3(bvh.aabbMaxX, bvh.aabbMaxY, bvh.aabbMaxZ);
		float distAABB = RayAABB(o, d, aabbMin, aabbMax);
		if (distAABB < mint && distAABB < maxdist)
		//if (distAABB != FLT_MAX)
		{
			float t = raycastRange_dist(o, d, bvh.start, bvh.end, mindist);
			if (t < mint)
			{
				mint = t;
			}
			++i;
		}
		else
		{
			i = bvh.jump;
		}
	}

	return mint;
}
