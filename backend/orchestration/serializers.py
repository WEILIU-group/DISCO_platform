from rest_framework import serializers
from .models import Job


class FileContentSerializer(serializers.Serializer):
    name = serializers.CharField()
    content = serializers.CharField()


class SubmitJobSerializer(serializers.Serializer):
    module = serializers.CharField()
    command = serializers.CharField()
    files = FileContentSerializer(many=True, required=False)
    slurm = serializers.DictField(required=False)
    remote_subdir = serializers.CharField(required=False, allow_blank=True)
    cluster = serializers.CharField(required=False, allow_blank=True)


class JobSerializer(serializers.ModelSerializer):
    class Meta:
        model = Job
        fields = [
            "id",
            "module",
            "cluster",
            "remote_dir",
            "job_id",
            "status",
            "payload",
            "created_at",
            "updated_at",
        ]
