# Generated by Django 5.1.6 on 2025-04-20 14:14

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0021_alter_mytrainedmodels_utilisateur'),
    ]

    operations = [
        migrations.CreateModel(
            name='AnalyseResult',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('model_name', models.CharField(max_length=100)),
                ('prediction', models.CharField(max_length=25)),
                ('proba', models.FloatField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('input_data', models.JSONField()),
                ('utilisateur', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
