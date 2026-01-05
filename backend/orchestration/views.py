from rest_framework import status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from .models import Job
from .serializers import JobSerializer, SubmitJobSerializer
from .services import fetch_status, submit_job


class JobViewSet(viewsets.ModelViewSet):
    queryset = Job.objects.all()
    serializer_class = JobSerializer
    http_method_names = ["get", "post", "patch", "head", "options"]

    def create(self, request, *args, **kwargs):
        payload = SubmitJobSerializer(data=request.data)
        payload.is_valid(raise_exception=True)
        data = payload.validated_data
        try:
            job_id, remote_dir = submit_job(
                module=data["module"],
                command=data["command"],
                files=data.get("files"),
                slurm_overrides=data.get("slurm"),
                remote_subdir=data.get("remote_subdir", ""),
                cluster_name=data.get("cluster"),
            )
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        job = Job.objects.create(
            module=data["module"],
            cluster=data.get("cluster", "default"),
            remote_dir=remote_dir,
            job_id=job_id,
            status="SUBMITTED",
            payload=data,
        )
        return Response(JobSerializer(job).data, status=status.HTTP_201_CREATED)

    @action(detail=True, methods=["post"], url_path="refresh")
    def refresh(self, request, pk=None):
        job = self.get_object()
        if not job.job_id:
            return Response({"detail": "job_id missing"}, status=status.HTTP_400_BAD_REQUEST)
        try:
            remote_status = fetch_status(job.job_id, job.cluster)
            job.status = remote_status
            job.save(update_fields=["status", "updated_at"])
        except Exception as exc:
            return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(JobSerializer(job).data)
